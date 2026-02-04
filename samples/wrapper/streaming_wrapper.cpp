// Streaming Wrapper for BOINC
// Based on BOINC wrapper.cpp
// Modified for streaming applications that need graceful suspend/resume
//
// Changes from standard wrapper:
// - Soft suspend: signals app via file, waits for graceful shutdown, then kills
// - Resume: restarts worker fresh
// - Designed for websocket-based streaming applications that can't be paused
//
// This file is part of BOINC.
// http://boinc.berkeley.edu
// Copyright (C) 2014 University of California
//
// BOINC is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.

#ifndef _WIN32
#include "config.h"
#endif
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>
#ifdef _WIN32
#include "boinc_win.h"
#include "win_util.h"
#else
#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif
#include <unistd.h>
#include <signal.h>
#endif

#include "boinc_api.h"
#include "app_ipc.h"
#include "error_numbers.h"
#include "filesys.h"
#include "parse.h"
#include "proc_control.h"
#include "procinfo.h"
#include "str_util.h"
#include "str_replace.h"
#include "util.h"

using std::vector;
using std::string;

#define JOB_FILENAME "job.xml"
#define CHECKPOINT_FILENAME "streaming_wrapper_checkpoint.txt"
#define SUSPEND_SIGNAL_FILE "suspend_signal"
#define POLL_PERIOD 1.0
#define SUSPEND_WAIT_TIME 2.0  // seconds to wait for graceful shutdown

int nthreads = 1;
int gpu_device_num = -1;
double runtime = 0;
APP_INIT_DATA aid;

struct TASK {
    string application;
    string exec_dir;
    vector<string> vsetenv;
    string stdin_filename;
    string stdout_filename;
    string stderr_filename;
    string checkpoint_filename;
    string fraction_done_filename;
    string command_line;
    double weight;
    bool is_daemon;
    bool append_cmdline_args;
    bool multi_process;
    double time_limit;
    int priority;

    // dynamic stuff
    double current_cpu_time;
    double final_cpu_time;
    double starting_cpu;
    bool suspended;
    bool killed_for_suspend;  // track if we killed for suspend vs real exit
    double elapsed_time;
#ifdef _WIN32
    HANDLE pid_handle;
    DWORD pid;
    struct _stat last_stat;
#else
    int pid;
    struct stat last_stat;
    double start_rusage;
#endif
    bool stat_first;

    int parse(XML_PARSER&);
    bool poll(int& status);
    int run(const vector<string> &child_args);
    void kill();
    void soft_stop();  // graceful stop for streaming apps
    void stop();
    void resume();
    double cpu_time();

    inline double fraction_done() {
        if (fraction_done_filename.size() == 0) return 0;
        FILE* f = fopen(fraction_done_filename.c_str(), "r");
        if (!f) return 0;
        fseek(f, -32, SEEK_END);
        double temp, frac = 0;
        while (!feof(f)) {
            char buf[256];
            char* p = fgets(buf, sizeof(buf), f);
            if (p == NULL) break;
            int n = sscanf(buf, "%lf", &temp);
            if (n == 1) frac = temp;
        }
        fclose(f);
        if (frac < 0) return 0;
        if (frac > 1) return 1;
        return frac;
    }

#ifdef _WIN32
    void set_up_env_vars(char** env_vars, const int nvars) {
        int bufsize = 0;
        for (int j = 0; j < nvars; j++) {
            bufsize += (1 + (int)vsetenv[j].length());
        }
        bufsize++;
        *env_vars = new char[bufsize];
        memset(*env_vars, 0, sizeof(char) * bufsize);
        char* p = *env_vars;
        int len = 0;
        for (vector<string>::iterator it = vsetenv.begin();
            it != vsetenv.end() && len < bufsize-1; ++it) {
            strncpy(p, it->c_str(), it->length());
            len = (int)strlen(p);
            p += len + 1;
        }
    }
#else
    void set_up_env_vars(char*** env_vars, const int nvars) {
        *env_vars = new char*[nvars+1];
        memset(*env_vars, 0x00, sizeof(char*) * (nvars+1));
        for (int i = 0; i < nvars; i++) {
            (*env_vars)[i] = const_cast<char*>(vsetenv[i].c_str());
        }
    }
#endif
};

vector<TASK> tasks;
vector<TASK> daemons;

// Write suspend signal file to notify app of pending suspend
void write_suspend_signal() {
    FILE* f = fopen(SUSPEND_SIGNAL_FILE, "w");
    if (f) {
        fprintf(f, "suspend\n");
        fclose(f);
    }
}

// Remove suspend signal file
void remove_suspend_signal() {
    boinc_delete_file(SUSPEND_SIGNAL_FILE);
}

// Check if worker acknowledged suspend (by deleting the signal file)
bool check_suspend_acknowledged() {
    return !boinc_file_exists(SUSPEND_SIGNAL_FILE);
}

void macro_substitute(string &str) {
    const char* pd = strlen(aid.project_dir) ? aid.project_dir : ".";
    char nt[256], cwd[1024];
    sprintf(nt, "%d", nthreads);
#ifdef _WIN32
    GetCurrentDirectory(sizeof(cwd), cwd);
#else
    getcwd(cwd, sizeof(cwd));
#endif

    // Simple string replacement
    size_t pos;
    while ((pos = str.find("$PROJECT_DIR")) != string::npos) {
        str.replace(pos, 12, pd);
    }
    while ((pos = str.find("$NTHREADS")) != string::npos) {
        str.replace(pos, 9, nt);
    }
    while ((pos = str.find("$PWD")) != string::npos) {
        str.replace(pos, 4, cwd);
    }
    if (gpu_device_num >= 0) {
        char gd[32];
        sprintf(gd, "%d", gpu_device_num);
        while ((pos = str.find("$GPU_DEVICE_NUM")) != string::npos) {
            str.replace(pos, 15, gd);
        }
    }
}

int TASK::parse(XML_PARSER& xp) {
    char buf[8192];
    weight = 1;
    current_cpu_time = 0;
    final_cpu_time = 0;
    stat_first = true;
    pid = 0;
    is_daemon = false;
    multi_process = false;
    append_cmdline_args = false;
    time_limit = 0;
    killed_for_suspend = false;
    priority = PROCESS_PRIORITY_LOWEST;

    while (!xp.get_tag()) {
        if (!xp.is_tag) continue;
        if (xp.match_tag("/task")) return 0;
        else if (xp.parse_string("application", application)) continue;
        else if (xp.parse_str("exec_dir", buf, sizeof(buf))) { exec_dir = buf; continue; }
        else if (xp.parse_str("setenv", buf, sizeof(buf))) { vsetenv.push_back(buf); continue; }
        else if (xp.parse_string("stdin_filename", stdin_filename)) continue;
        else if (xp.parse_string("stdout_filename", stdout_filename)) continue;
        else if (xp.parse_string("stderr_filename", stderr_filename)) continue;
        else if (xp.parse_str("command_line", buf, sizeof(buf))) { command_line = buf; continue; }
        else if (xp.parse_string("checkpoint_filename", checkpoint_filename)) continue;
        else if (xp.parse_string("fraction_done_filename", fraction_done_filename)) continue;
        else if (xp.parse_double("weight", weight)) continue;
        else if (xp.parse_bool("daemon", is_daemon)) continue;
        else if (xp.parse_bool("multi_process", multi_process)) continue;
        else if (xp.parse_bool("append_cmdline_args", append_cmdline_args)) continue;
        else if (xp.parse_double("time_limit", time_limit)) continue;
        else if (xp.parse_int("priority", priority)) continue;
    }
    return ERR_XML_PARSE;
}

int parse_job_file() {
    MIOFILE mf;
    char buf[256], buf2[256];

    boinc_resolve_filename(JOB_FILENAME, buf, sizeof(buf));
    FILE* f = boinc_fopen(buf, "r");
    if (!f) {
        fprintf(stderr, "%s can't open job file %s\n", boinc_msg_prefix(buf2, sizeof(buf2)), buf);
        return ERR_FOPEN;
    }
    mf.init_file(f);
    XML_PARSER xp(&mf);

    if (!xp.parse_start("job_desc")) return ERR_XML_PARSE;
    while (!xp.get_tag()) {
        if (!xp.is_tag) continue;
        if (xp.match_tag("/job_desc")) { fclose(f); return 0; }
        if (xp.match_tag("task")) {
            TASK task;
            int retval = task.parse(xp);
            if (!retval) {
                if (task.is_daemon) {
                    daemons.push_back(task);
                } else {
                    tasks.push_back(task);
                }
            }
            continue;
        }
    }
    fclose(f);
    return ERR_XML_PARSE;
}

#ifdef _WIN32
HANDLE win_fopen(const char* path, const char* mode) {
    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    if (!strcmp(mode, "r")) {
        return CreateFile(path, GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, 0, 0);
    } else if (!strcmp(mode, "w")) {
        return CreateFile(path, GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, &sa, OPEN_ALWAYS, 0, 0);
    } else if (!strcmp(mode, "a")) {
        HANDLE hAppend = CreateFile(path, GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, &sa, OPEN_ALWAYS, 0, 0);
        SetFilePointer(hAppend, 0, NULL, FILE_END);
        return hAppend;
    }
    return 0;
}
#endif

int TASK::run(const vector<string> &child_args) {
    string stdout_path, stdin_path, stderr_path;
    char app_path[1024], buf[256];

    if (fraction_done_filename.size()) {
        boinc_delete_file(fraction_done_filename.c_str());
    }

    // Remove any stale suspend signal
    remove_suspend_signal();
    killed_for_suspend = false;

    strcpy(buf, application.c_str());
    char* p = strstr(buf, "$PROJECT_DIR");
    if (p) {
        p += strlen("$PROJECT_DIR");
        sprintf(app_path, "%s%s", aid.project_dir, p);
    } else {
        boinc_resolve_filename(buf, app_path, sizeof(app_path));
    }

    if (!boinc_file_exists(app_path)) {
        fprintf(stderr, "application %s missing\n", app_path);
        return ERR_NOT_FOUND;
    }

    if (append_cmdline_args) {
        for (const string& arg: child_args) {
            command_line += string(" ") + arg;
        }
    }

    fprintf(stderr, "%s Streaming wrapper: running %s (%s)\n",
        boinc_msg_prefix(buf, sizeof(buf)), app_path, command_line.c_str());

    int priority_val = 0;
    if (!aid.no_priority_change) {
        if (aid.process_priority > CONFIG_PRIORITY_UNSPECIFIED) {
            priority_val = process_priority_value(aid.process_priority+1);
        } else {
            priority_val = process_priority_value(priority);
        }
    }

#ifdef _WIN32
    PROCESS_INFORMATION process_info;
    STARTUPINFO startup_info;
    string command;

    memset(&process_info, 0, sizeof(process_info));
    memset(&startup_info, 0, sizeof(startup_info));

    command = string("\"") + app_path + string("\" ") + command_line;

    startup_info.dwFlags = STARTF_USESTDHANDLES;
    if (stdout_filename != "") {
        boinc_resolve_filename_s(stdout_filename.c_str(), stdout_path);
        startup_info.hStdOutput = win_fopen(stdout_path.c_str(), "a");
    } else {
        startup_info.hStdOutput = (HANDLE)_get_osfhandle(_fileno(stderr));
    }
    if (stdin_filename != "") {
        boinc_resolve_filename_s(stdin_filename.c_str(), stdin_path);
        startup_info.hStdInput = win_fopen(stdin_path.c_str(), "r");
    }
    if (stderr_filename != "") {
        boinc_resolve_filename_s(stderr_filename.c_str(), stderr_path);
        startup_info.hStdError = win_fopen(stderr_path.c_str(), "a");
    } else {
        startup_info.hStdError = (HANDLE)_get_osfhandle(_fileno(stderr));
    }

    int nvars = (int)vsetenv.size();
    char* env_vars = NULL;
    if (nvars > 0) {
        set_up_env_vars(&env_vars, nvars);
    }

    BOOL success = CreateProcess(
        NULL, (LPSTR)command.c_str(), NULL, NULL, TRUE,
        CREATE_NO_WINDOW|priority_val,
        (LPVOID)env_vars,
        exec_dir.empty() ? NULL : exec_dir.c_str(),
        &startup_info, &process_info
    );

    if (!success) {
        char error_msg[1024];
        windows_format_error_string(GetLastError(), error_msg, sizeof(error_msg));
        fprintf(stderr, "can't run app: %s\n", error_msg);
        if (env_vars) delete[] env_vars;
        return ERR_EXEC;
    }
    if (env_vars) delete[] env_vars;
    pid_handle = process_info.hProcess;
    pid = process_info.dwProcessId;
    CloseHandle(process_info.hThread);
#else
    char* argv[256];
    char arglist[4096];

    struct rusage ru;
    getrusage(RUSAGE_CHILDREN, &ru);
    start_rusage = (float)ru.ru_utime.tv_sec + ((float)ru.ru_utime.tv_usec)/1e+6;

    pid = fork();
    if (pid == -1) {
        perror("fork(): ");
        return ERR_FORK;
    }
    if (pid == 0) {
        if (stdout_filename != "") {
            boinc_resolve_filename_s(stdout_filename.c_str(), stdout_path);
            freopen(stdout_path.c_str(), "a", stdout);
        }
        if (stdin_filename != "") {
            boinc_resolve_filename_s(stdin_filename.c_str(), stdin_path);
            freopen(stdin_path.c_str(), "r", stdin);
        }
        if (stderr_filename != "") {
            boinc_resolve_filename_s(stderr_filename.c_str(), stderr_path);
            freopen(stderr_path.c_str(), "a", stderr);
        }

        argv[0] = app_path;
        strlcpy(arglist, command_line.c_str(), sizeof(arglist));
        parse_command_line(arglist, argv+1);

        if (priority_val) {
            setpriority(PRIO_PROCESS, 0, priority_val);
        }
        if (!exec_dir.empty()) {
            chdir(exec_dir.c_str());
        }

        const int nvars = vsetenv.size();
        char** env_vars = NULL;
        if (nvars > 0) {
            set_up_env_vars(&env_vars, nvars);
            execve(app_path, argv, env_vars);
        } else {
            execv(app_path, argv);
        }
        perror("execv() failed: ");
        exit(ERR_EXEC);
    }
#endif

    fprintf(stderr, "%s Streaming wrapper: created child process %d\n",
        boinc_msg_prefix(buf, sizeof(buf)), (int)pid);

    suspended = false;
    elapsed_time = 0;
    return 0;
}

bool TASK::poll(int& status) {
    char buf[256];

    if (time_limit && elapsed_time > time_limit) {
        fprintf(stderr, "%s task reached time limit %.0f\n",
            boinc_msg_prefix(buf, sizeof(buf)), time_limit);
        kill();
        status = 0;
        return true;
    }

#ifdef _WIN32
    unsigned long exit_code;
    if (GetExitCodeProcess(pid_handle, &exit_code)) {
        if (exit_code != STILL_ACTIVE) {
            // If we killed it for suspend, don't treat as real exit
            if (killed_for_suspend) {
                return false;
            }
            status = exit_code;
            final_cpu_time = current_cpu_time;
            fprintf(stderr, "%s %s exited; CPU time %f\n",
                boinc_msg_prefix(buf, sizeof(buf)), application.c_str(), final_cpu_time);
            CloseHandle(pid_handle);
            return true;
        }
    }
#else
    int wpid;
    struct rusage ru;

    wpid = waitpid(pid, &status, WNOHANG);
    if (wpid) {
        if (killed_for_suspend) {
            return false;
        }
        getrusage(RUSAGE_CHILDREN, &ru);
        final_cpu_time = (float)ru.ru_utime.tv_sec + ((float)ru.ru_utime.tv_usec)/1e+6;
        final_cpu_time -= start_rusage;
        fprintf(stderr, "%s %s exited; CPU time %f\n",
            boinc_msg_prefix(buf, sizeof(buf)), application.c_str(), final_cpu_time);
        if (WIFEXITED(status)) {
            status = WEXITSTATUS(status);
        }
        return true;
    }
#endif
    return false;
}

void TASK::kill() {
    char buf[256];
    fprintf(stderr, "%s Streaming wrapper: killing task\n", boinc_msg_prefix(buf, sizeof(buf)));
#ifdef _WIN32
    kill_descendants();
#else
    kill_descendants(pid);
#endif
}

// Soft stop - signal the app to shut down gracefully, then kill
// This allows streaming apps to close websocket connections cleanly
void TASK::soft_stop() {
    char buf[256];
    fprintf(stderr, "%s Streaming wrapper: soft stopping task (signaling graceful shutdown)\n",
        boinc_msg_prefix(buf, sizeof(buf)));

    // Write suspend signal file
    write_suspend_signal();

    // Wait for app to acknowledge (up to SUSPEND_WAIT_TIME seconds)
    double start = dtime();
    while (dtime() - start < SUSPEND_WAIT_TIME) {
        if (check_suspend_acknowledged()) {
            fprintf(stderr, "%s Streaming wrapper: app acknowledged suspend\n",
                boinc_msg_prefix(buf, sizeof(buf)));
            break;
        }
        boinc_sleep(0.1);
    }

    // Kill the child process tree
    kill();

    // Mark that we killed for suspend (not a real exit)
    killed_for_suspend = true;
    suspended = true;

    fprintf(stderr, "%s Streaming wrapper: soft stop complete\n", boinc_msg_prefix(buf, sizeof(buf)));
}

void TASK::stop() {
    // Use soft stop instead of hard suspend for streaming apps
    soft_stop();
}

void TASK::resume() {
    char buf[256];
    fprintf(stderr, "%s Streaming wrapper: resuming task\n", boinc_msg_prefix(buf, sizeof(buf)));

    // Remove suspend signal
    remove_suspend_signal();

    // Restart the worker
    killed_for_suspend = false;
    suspended = false;

    vector<string> empty_args;
    int retval = run(empty_args);
    if (retval) {
        fprintf(stderr, "%s Streaming wrapper: failed to restart task: %d\n",
            boinc_msg_prefix(buf, sizeof(buf)), retval);
    }
}

double TASK::cpu_time() {
#ifndef ANDROID
    double x = process_tree_cpu_time(pid);
    if (x > current_cpu_time) {
        current_cpu_time = x;
    }
#endif
    return current_cpu_time;
}

void poll_boinc_messages(TASK& task) {
    BOINC_STATUS status;
    boinc_get_status(&status);

    if (status.no_heartbeat || status.quit_request || status.abort_request) {
        char buf[256];
        fprintf(stderr, "%s Streaming wrapper: received quit/abort\n", boinc_msg_prefix(buf, sizeof(buf)));
        task.kill();
        exit(0);
    }

    // Suspend handling for streaming apps
    // Kill the worker, wait for resume, then restart
    if (status.suspended && !task.suspended) {
        char buf[256];
        fprintf(stderr, "%s Streaming wrapper: suspended - killing worker and waiting\n", boinc_msg_prefix(buf, sizeof(buf)));
        task.kill();
        task.suspended = true;
    }
}

void write_checkpoint(int ntasks_completed, double cpu, double rt) {
    boinc_begin_critical_section();
    FILE* f = fopen(CHECKPOINT_FILENAME, "w");
    if (f) {
        fprintf(f, "%d %f %f\n", ntasks_completed, cpu, rt);
        fclose(f);
    }
    boinc_checkpoint_completed();
}

int read_checkpoint(int& ntasks_completed, double& cpu, double& rt) {
    ntasks_completed = 0;
    cpu = 0;
    rt = 0;
    FILE* f = fopen(CHECKPOINT_FILENAME, "r");
    if (!f) return ERR_FOPEN;
    int n = fscanf(f, "%d %lf %lf", &ntasks_completed, &cpu, &rt);
    fclose(f);
    if (n != 3) return -1;
    return 0;
}

int main(int argc, char** argv) {
    BOINC_OPTIONS options;
    int retval, ntasks_completed;
    double total_weight = 0, weight_completed = 0;
    double checkpoint_cpu_time;
    char buf[256];
    vector<string> child_args;

    fprintf(stderr, "%s BOINC streaming wrapper starting\n", boinc_msg_prefix(buf, sizeof(buf)));

#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
#endif

    // Parse command line
    for (int j = 1; j < argc; j++) {
        if (!strcmp(argv[j], "--nthreads")) {
            nthreads = atoi(argv[++j]);
        } else if (!strcmp(argv[j], "--device")) {
            gpu_device_num = atoi(argv[++j]);
        } else {
            child_args.push_back(argv[j]);
        }
    }

    // MUST initialize BOINC before parse_job_file() because
    // boinc_resolve_filename() needs the API to be initialized
    memset(&options, 0, sizeof(options));
    options.main_program = true;
    options.check_heartbeat = true;
    options.handle_process_control = true;

    boinc_init_options(&options);
    boinc_get_init_data(aid);

    retval = parse_job_file();
    if (retval) {
        fprintf(stderr, "%s can't parse job file: %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval);
        boinc_finish(retval);
    }

    retval = read_checkpoint(ntasks_completed, checkpoint_cpu_time, runtime);
    if (retval) {
        write_checkpoint(0, 0, 0);
    }

    for (unsigned int i = 0; i < tasks.size(); i++) {
        total_weight += tasks[i].weight;
        macro_substitute(tasks[i].application);
        macro_substitute(tasks[i].command_line);
        macro_substitute(tasks[i].exec_dir);
    }

    // Main task loop
    for (unsigned int i = 0; i < tasks.size(); i++) {
        TASK& task = tasks[i];
        if ((int)i < ntasks_completed) {
            weight_completed += task.weight;
            continue;
        }

        double frac_done = weight_completed / total_weight;
        double cpu_time = 0;

        task.starting_cpu = checkpoint_cpu_time;
        retval = task.run(child_args);

        if (retval) {
            fprintf(stderr, "%s Streaming wrapper: task.run() failed: %d\n",
                boinc_msg_prefix(buf, sizeof(buf)), retval);
            boinc_finish(retval);
        }

        int counter = 0;
        while (1) {
            int status;
            if (task.poll(status)) {
                // Task exited - check if it was due to suspend (we killed it)
                if (task.suspended) {
                    // Wait for BOINC to resume us
                    fprintf(stderr, "%s Streaming wrapper: waiting for resume\n",
                        boinc_msg_prefix(buf, sizeof(buf)));
                    while (task.suspended) {
                        BOINC_STATUS bstatus;
                        boinc_get_status(&bstatus);

                        if (bstatus.no_heartbeat || bstatus.quit_request || bstatus.abort_request) {
                            exit(0);
                        }

                        if (!bstatus.suspended) {
                            // Resume! Restart the worker
                            fprintf(stderr, "%s Streaming wrapper: resuming - restarting worker\n",
                                boinc_msg_prefix(buf, sizeof(buf)));
                            task.suspended = false;
                            retval = task.run(child_args);
                            if (retval) {
                                fprintf(stderr, "%s Streaming wrapper: task.run() on resume failed: %d\n",
                                    boinc_msg_prefix(buf, sizeof(buf)), retval);
                                boinc_finish(retval);
                            }
                        }

                        boinc_sleep(POLL_PERIOD);
                    }
                    continue;  // back to main poll loop
                }

                // Normal exit
                if (status) {
                    fprintf(stderr, "%s Streaming wrapper: app exit status: 0x%x\n",
                        boinc_msg_prefix(buf, sizeof(buf)), status);
                    boinc_finish(EXIT_CHILD_FAILED);
                }
                break;
            }

            poll_boinc_messages(task);

            double task_fraction_done = task.fraction_done();
            double delta = task_fraction_done * task.weight / total_weight;

            if (counter % 10 == 0) {
                cpu_time = task.cpu_time();
            }

            boinc_report_app_status(
                task.starting_cpu + cpu_time,
                checkpoint_cpu_time,
                frac_done + delta
            );

            boinc_sleep(POLL_PERIOD);
            if (!task.suspended) {
                task.elapsed_time += POLL_PERIOD;
                runtime += POLL_PERIOD;
            }
            counter++;
        }

        checkpoint_cpu_time = task.starting_cpu + task.final_cpu_time;
        write_checkpoint(i + 1, checkpoint_cpu_time, runtime);
        weight_completed += task.weight;
    }

    boinc_finish(0);
    return 0;
}
