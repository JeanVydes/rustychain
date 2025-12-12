//! OS Command execution tool for agents.
//!
//! Allows agents to execute shell commands with full interactive support,
//! including handling of password prompts, confirmations, and user input.

use crate::llm::function::{FnExecutor, ToolArgs};
use crate::{FnDeclarator, FunctionDeclaration};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Patterns that indicate the command is waiting for user input.
const INPUT_PATTERNS: &[&str] = &[
    "password:",
    "passphrase:",
    "[y/n]",
    "(y/n)",
    "yes/no",
    "[yes/no]",
    "(yes/no)",
    "continue?",
    "proceed?",
    "confirm",
    "enter",
    "input:",
    "prompt:",
    "username:",
    "user:",
    "login:",
    "token:",
    "api key:",
    "secret:",
    "[sudo]",
    "are you sure",
    "overwrite",
    "replace",
    "delete",
    "(y/n/a)",
    "[y/n/a]",
];

/// Action to take when a command requests input.
#[derive(Debug, Clone)]
pub enum InputAction {
    /// Provide the input text to the command.
    Provide(String),
    /// Cancel/kill the command.
    Cancel,
    /// Skip this input prompt (send empty string).
    Skip,
    /// Send Ctrl+C (SIGINT) to the process.
    Interrupt,
    /// Send EOF (Ctrl+D) to close stdin.
    SendEof,
}

/// Context provided to the input callback with all available information.
#[derive(Debug, Clone)]
pub struct InputContext {
    /// The prompt/output that triggered the input request.
    pub prompt: String,
    /// All stdout collected so far.
    pub stdout_so_far: String,
    /// All stderr collected so far.
    pub stderr_so_far: String,
    /// The original command being executed.
    pub command: String,
    /// The arguments passed to the command.
    pub args: Vec<String>,
    /// Time elapsed in milliseconds since command started.
    pub elapsed_ms: u64,
    /// Which input pattern was matched (if any).
    pub matched_pattern: Option<String>,
}

/// An environment variable key-value pair.
#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct EnvVar {
    /// The environment variable name.
    #[schemars(description = "The environment variable name.")]
    pub name: String,
    /// The environment variable value.
    #[schemars(description = "The environment variable value.")]
    pub value: String,
}

/// Arguments for executing a command.
#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct CommandArgs {
    /// The command to execute (e.g., "ls", "git", "cargo").
    #[schemars(description = "The command to execute (e.g., 'ls', 'git', 'cargo').")]
    pub command: String,
    /// Arguments to pass to the command.
    #[schemars(description = "Arguments to pass to the command. (e.g., ['-la', '/home'])")]
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory for the command. If not provided, uses current directory.
    #[schemars(
        description = "Working directory for the command. If not provided, uses current directory."
    )]
    #[serde(default)]
    pub working_dir: Option<String>,
    /// Environment variables to set for the command as a list of name-value pairs.
    #[schemars(
        description = "Environment variables to set for the command as a list of name-value pairs."
    )]
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// Timeout in seconds. Default is 300 (5 minutes). Set to 0 for no timeout.
    #[schemars(
        description = "Timeout in seconds. Default is 300 (5 minutes). Set to 0 for no timeout."
    )]
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Run the command in the background. Returns immediately with a process ID.
    #[schemars(
        description = "Run the command in the background. Returns immediately with a process ID."
    )]
    #[serde(default)]
    pub background: bool,
    /// Use shell to execute the command (allows pipes, redirects, etc.)
    #[schemars(description = "Use shell to execute the command (allows pipes, redirects, etc.)")]
    #[serde(default)]
    pub use_shell: bool,
    /// Custom input patterns to detect (in addition to defaults).
    #[schemars(description = "Custom input patterns to detect (in addition to defaults).")]
    #[serde(default)]
    pub custom_input_patterns: Vec<String>,
}

fn default_timeout() -> u64 {
    300
}

impl ToolArgs for CommandArgs {}

/// Result of a command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    /// The stdout output of the command.
    pub stdout: String,
    /// The stderr output of the command.
    pub stderr: String,
    /// The exit code of the command.
    pub exit_code: Option<i32>,
    /// Whether the command timed out.
    pub timed_out: bool,
    /// Whether the command was killed by user/callback.
    pub killed: bool,
    /// Duration of the command in milliseconds.
    pub duration_ms: u64,
    /// Reason for cancellation if killed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel_reason: Option<String>,
}

/// Callback type for when the command needs user input.
/// Receives full context and returns an action to take.
pub type InputCallback = Box<
    dyn Fn(
            InputContext,
        ) -> Pin<Box<dyn std::future::Future<Output = crate::Result<InputAction>> + Send>>
        + Send
        + Sync,
>;

/// A tool that executes OS commands with full interactive support.
#[derive(Clone)]
pub struct CommandTool {
    /// Custom input callback. If None, reads from stdin.
    input_callback: Option<Arc<InputCallback>>,
    /// Additional input patterns to detect.
    additional_patterns: Vec<String>,
}

impl Default for CommandTool {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandTool {
    /// Create a new CommandTool with default settings.
    pub fn new() -> Self {
        Self {
            input_callback: None,
            additional_patterns: vec![],
        }
    }

    /// Set a custom callback for handling input requests.
    ///
    /// The callback receives an `InputContext` with full information about
    /// the command execution state and returns an `InputAction` to decide
    /// what to do.
    pub fn with_input_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(InputContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = crate::Result<InputAction>> + Send + 'static,
    {
        self.input_callback = Some(Arc::new(Box::new(move |ctx: InputContext| {
            let fut = callback(ctx);
            Box::pin(fut)
                as Pin<Box<dyn std::future::Future<Output = crate::Result<InputAction>> + Send>>
        })));
        self
    }

    /// Add additional patterns to detect input prompts.
    pub fn with_additional_patterns(mut self, patterns: Vec<String>) -> Self {
        self.additional_patterns = patterns;
        self
    }

    /// Check if the output indicates the command is waiting for input.
    /// Returns the matched pattern if found.
    fn find_input_pattern(&self, line: &str, custom_patterns: &[String]) -> Option<String> {
        let lower = line.to_lowercase();

        // Check default patterns
        for pattern in INPUT_PATTERNS {
            if lower.contains(pattern) {
                return Some(pattern.to_string());
            }
        }

        // Check additional patterns from tool config
        for pattern in &self.additional_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return Some(pattern.clone());
            }
        }

        // Check custom patterns from args
        for pattern in custom_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return Some(pattern.clone());
            }
        }

        None
    }

    /// Check if the output indicates the command is waiting for input.
    pub fn needs_input(&self, line: &str, custom_patterns: &[String]) -> bool {
        self.find_input_pattern(line, custom_patterns).is_some()
    }

    /// Get input action from the callback or terminal.
    async fn get_input_action(&self, context: InputContext) -> crate::Result<InputAction> {
        if let Some(ref callback) = self.input_callback {
            callback(context).await
        } else {
            // Default: ask user via terminal
            let input = get_user_input_from_terminal(&context.prompt).await?;
            Ok(InputAction::Provide(input))
        }
    }

    /// Build the command with all options.
    fn build_command(&self, args: &CommandArgs) -> Command {
        let mut cmd = if args.use_shell {
            let shell = if cfg!(target_os = "windows") {
                "cmd"
            } else {
                "sh"
            };
            let shell_arg = if cfg!(target_os = "windows") {
                "/C"
            } else {
                "-c"
            };

            let full_cmd = if args.args.is_empty() {
                args.command.clone()
            } else {
                format!("{} {}", args.command, args.args.join(" "))
            };

            let mut c = Command::new(shell);
            c.arg(shell_arg).arg(full_cmd);
            c
        } else {
            let mut c = Command::new(&args.command);
            c.args(&args.args);
            c
        };

        // Set working directory
        if let Some(ref dir) = args.working_dir {
            cmd.current_dir(dir);
        }

        // Set environment variables
        for env_var in &args.env {
            cmd.env(&env_var.name, &env_var.value);
        }

        // Configure stdio
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        cmd
    }

    /// Execute the command and handle I/O.
    async fn execute_command(&self, args: CommandArgs) -> crate::Result<CommandResult> {
        use tokio::sync::mpsc;

        let start_time = std::time::Instant::now();

        let mut cmd = self.build_command(&args);

        info!("Executing command: {} {:?}", args.command, args.args);

        let mut child = cmd.spawn().map_err(|e| {
            crate::CoreError::Generic(format!("Failed to spawn command '{}': {}", args.command, e))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| crate::CoreError::Generic("Failed to capture stdin".to_string()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| crate::CoreError::Generic("Failed to capture stdout".to_string()))?;

        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| crate::CoreError::Generic("Failed to capture stderr".to_string()))?;

        let stdin_writer = Arc::new(Mutex::new(Some(stdin)));
        let mut stdout_reader = BufReader::new(stdout);
        let mut stderr_reader = BufReader::new(stderr);

        let full_stdout = Arc::new(Mutex::new(String::new()));
        let full_stderr = Arc::new(Mutex::new(String::new()));
        let cancelled = Arc::new(Mutex::new(Option::<String>::None));

        let custom_patterns = args.custom_input_patterns.clone();
        let command_str = args.command.clone();
        let args_vec = args.args.clone();
        let timeout_duration = if args.timeout_secs > 0 {
            Some(Duration::from_secs(args.timeout_secs))
        } else {
            None
        };

        // Channel to send input prompts to main task
        let (prompt_tx, mut prompt_rx) = mpsc::channel::<(String, String, Option<String>)>(16); // (source, line, pattern)

        // Clone for tasks
        let stdout_output = Arc::clone(&full_stdout);
        let stderr_output = Arc::clone(&full_stderr);
        let patterns_for_stdout = custom_patterns.clone();
        let patterns_for_stderr = custom_patterns.clone();
        let tool_for_stdout = self.clone();
        let tool_for_stderr = self.clone();
        let prompt_tx_stdout = prompt_tx.clone();
        let prompt_tx_stderr = prompt_tx;

        // Spawn stdout reader task
        let stdout_handle = tokio::spawn(async move {
            loop {
                let mut line = String::new();
                match stdout_reader.read_line(&mut line).await {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        debug!("stdout: {}", line.trim_end());
                        stdout_output.lock().await.push_str(&line);

                        // Check if we need input
                        if let Some(pattern) =
                            tool_for_stdout.find_input_pattern(&line, &patterns_for_stdout)
                        {
                            let _ = prompt_tx_stdout
                                .send(("stdout".to_string(), line, Some(pattern)))
                                .await;
                        }
                    }
                    Err(e) => {
                        warn!("stdout read error: {}", e);
                        break;
                    }
                }
            }
        });

        // Spawn stderr reader task
        let stderr_handle = tokio::spawn(async move {
            loop {
                let mut line = String::new();
                match stderr_reader.read_line(&mut line).await {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        debug!("stderr: {}", line.trim_end());
                        stderr_output.lock().await.push_str(&line);

                        // Check stderr for input prompts too
                        if let Some(pattern) =
                            tool_for_stderr.find_input_pattern(&line, &patterns_for_stderr)
                        {
                            let _ = prompt_tx_stderr
                                .send(("stderr".to_string(), line, Some(pattern)))
                                .await;
                        }
                    }
                    Err(e) => {
                        warn!("stderr read error: {}", e);
                        break;
                    }
                }
            }
        });

        // Main loop - handle input prompts and wait for process
        let stdin_for_loop = Arc::clone(&stdin_writer);
        let cancelled_for_loop = Arc::clone(&cancelled);
        let full_stdout_for_loop = Arc::clone(&full_stdout);
        let full_stderr_for_loop = Arc::clone(&full_stderr);

        let process_loop = async {
            loop {
                tokio::select! {
                    biased;

                    // Handle input prompts from reader tasks
                    Some((source, line, pattern)) = prompt_rx.recv() => {
                        info!("Detected input prompt in {}: {:?}", source, pattern);

                        // Build context for callback
                        let context = InputContext {
                            prompt: line.clone(),
                            stdout_so_far: full_stdout_for_loop.lock().await.clone(),
                            stderr_so_far: full_stderr_for_loop.lock().await.clone(),
                            command: command_str.clone(),
                            args: args_vec.clone(),
                            elapsed_ms: start_time.elapsed().as_millis() as u64,
                            matched_pattern: pattern,
                        };

                        // Get action from callback
                        match self.get_input_action(context).await {
                            Ok(action) => {
                                match action {
                                    InputAction::Provide(input) => {
                                        let mut guard = stdin_for_loop.lock().await;
                                        if let Some(ref mut stdin) = *guard {
                                            let input_with_newline = if input.ends_with('\n') {
                                                input
                                            } else {
                                                format!("{}\n", input)
                                            };
                                            if let Err(e) = stdin.write_all(input_with_newline.as_bytes()).await {
                                                warn!("Failed to write input: {}", e);
                                            }
                                            if let Err(e) = stdin.flush().await {
                                                warn!("Failed to flush stdin: {}", e);
                                            }
                                        }
                                    }
                                    InputAction::Cancel => {
                                        info!("Callback requested cancellation");
                                        *cancelled_for_loop.lock().await = Some("Cancelled by callback".to_string());
                                        let _ = child.kill().await;
                                        break;
                                    }
                                    InputAction::Skip => {
                                        debug!("Skipping input prompt");
                                        // Do nothing, just continue
                                    }
                                    InputAction::Interrupt => {
                                        info!("Sending interrupt signal");
                                        #[cfg(unix)]
                                        {
                                            if let Some(pid) = child.id() {
                                                let _ = std::process::Command::new("kill")
                                                    .args(["-2", &pid.to_string()])
                                                    .output();
                                            }
                                        }
                                        #[cfg(not(unix))]
                                        {
                                            let _ = child.kill().await;
                                        }
                                    }
                                    InputAction::SendEof => {
                                        info!("Closing stdin (EOF)");
                                        let mut guard = stdin_for_loop.lock().await;
                                        *guard = None; // Drop stdin to send EOF
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to get input action: {}", e);
                            }
                        }
                    }

                    // Check if process has exited
                    status = child.wait() => {
                        return status;
                    }
                }
            }

            // If we broke out of loop due to cancellation
            child.wait().await
        };

        // Wait for process with optional timeout
        let wait_result = if let Some(dur) = timeout_duration {
            match timeout(dur, process_loop).await {
                Ok(result) => result,
                Err(_) => {
                    warn!("Command timed out after {} seconds", args.timeout_secs);
                    let _ = child.kill().await;

                    // Wait for reader tasks
                    let _ = stdout_handle.await;
                    let _ = stderr_handle.await;

                    return Ok(CommandResult {
                        stdout: full_stdout.lock().await.clone(),
                        stderr: full_stderr.lock().await.clone(),
                        exit_code: None,
                        timed_out: true,
                        killed: true,
                        duration_ms: start_time.elapsed().as_millis() as u64,
                        cancel_reason: None,
                    });
                }
            }
        } else {
            process_loop.await
        };

        // Wait for reader tasks to complete
        let _ = stdout_handle.await;
        let _ = stderr_handle.await;

        let was_cancelled = cancelled.lock().await.clone();

        match wait_result {
            Ok(status) => Ok(CommandResult {
                stdout: full_stdout.lock().await.clone(),
                stderr: full_stderr.lock().await.clone(),
                exit_code: status.code(),
                timed_out: false,
                killed: was_cancelled.is_some(),
                duration_ms: start_time.elapsed().as_millis() as u64,
                cancel_reason: was_cancelled,
            }),
            Err(e) => Err(Box::from(crate::CoreError::Generic(format!(
                "Failed to wait for command: {}",
                e
            )))),
        }
    }
}

#[async_trait::async_trait]
impl FnExecutor<CommandArgs, serde_json::Value> for CommandTool {
    async fn call(&self, args: CommandArgs) -> crate::Result<serde_json::Value> {
        // Handle background execution
        if args.background {
            let mut cmd = self.build_command(&args);
            let child = cmd.spawn().map_err(|e| {
                crate::CoreError::Generic(format!(
                    "Failed to spawn background command '{}': {}",
                    args.command, e
                ))
            })?;

            let pid = child.id();
            info!("Started background process with PID: {:?}", pid);

            return Ok(serde_json::json!({
                "status": "background",
                "pid": pid,
                "message": format!("Command started in background with PID {:?}", pid)
            }));
        }

        // Execute command and wait for result
        let result = self.execute_command(args).await?;

        if result.timed_out {
            Ok(serde_json::json!({
                "status": "timeout",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
                "message": "Command timed out"
            }))
        } else if result.killed {
            Ok(serde_json::json!({
                "status": "cancelled",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "message": result.cancel_reason.unwrap_or_else(|| "Command was cancelled".to_string())
            }))
        } else if result.exit_code == Some(0) {
            Ok(serde_json::json!({
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms
            }))
        } else {
            Ok(serde_json::json!({
                "status": "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "message": format!("Command exited with code {:?}", result.exit_code)
            }))
        }
    }
}

/// Get user input from the terminal.
async fn get_user_input_from_terminal(prompt: &str) -> crate::Result<String> {
    // Print the prompt
    print!("\n🔒 Input required: {}\n> ", prompt.trim());
    tokio::io::stdout()
        .flush()
        .await
        .map_err(|e| crate::CoreError::Generic(format!("Failed to flush stdout: {}", e)))?;

    // Read input from stdin (blocking)
    tokio::task::spawn_blocking(|| {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| crate::CoreError::Generic(format!("Failed to read user input: {}", e)))?;
        Ok(input)
    })
    .await
    .map_err(|e| crate::CoreError::Generic(format!("Failed to join input task: {}", e)))?
}

/// Arguments for sending a signal to a process.
#[derive(JsonSchema, Serialize, Deserialize, Debug, Clone)]
pub struct SignalArgs {
    /// Process ID to send the signal to.
    #[schemars(description = "Process ID to send the signal to.")]
    pub pid: u32,
    /// Signal to send: "kill", "term", "int", "stop", "cont".
    #[schemars(description = "Signal to send: 'kill', 'term', 'int', 'stop', 'cont'.")]
    pub signal: String,
}

impl ToolArgs for SignalArgs {}

/// Tool for sending signals to processes.
#[derive(Clone, Default)]
pub struct SignalTool {}

#[async_trait::async_trait]
impl FnExecutor<SignalArgs, serde_json::Value> for SignalTool {
    async fn call(&self, args: SignalArgs) -> crate::Result<serde_json::Value> {
        #[cfg(unix)]
        {
            // Map signal names to their numeric values
            let signal_num = match args.signal.to_lowercase().as_str() {
                "kill" | "sigkill" | "9" => "9",
                "term" | "sigterm" | "15" => "15",
                "int" | "sigint" | "2" => "2",
                "stop" | "sigstop" | "19" => "19",
                "cont" | "sigcont" | "18" => "18",
                "hup" | "sighup" | "1" => "1",
                "usr1" | "sigusr1" | "10" => "10",
                "usr2" | "sigusr2" | "12" => "12",
                _ => {
                    return Err(Box::from(crate::CoreError::Generic(format!(
                        "Unknown signal: {}",
                        args.signal
                    ))));
                }
            };

            // Use the kill command to send signals
            let output = std::process::Command::new("kill")
                .arg(format!("-{}", signal_num))
                .arg(args.pid.to_string())
                .output()
                .map_err(|e| {
                    crate::CoreError::Generic(format!("Failed to execute kill command: {}", e))
                })?;

            if output.status.success() {
                Ok(serde_json::json!({
                    "status": "success",
                    "message": format!("Sent {} to process {}", args.signal, args.pid)
                }))
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(Box::from(crate::CoreError::Generic(format!(
                    "Failed to send signal {} to PID {}: {}",
                    args.signal, args.pid, stderr
                ))))
            }
        }

        #[cfg(not(unix))]
        {
            Err(Box::from(crate::CoreError::Generic(
                "Signal sending is only supported on Unix systems".to_string(),
            )))
        }
    }
}

impl FnDeclarator<CommandArgs, serde_json::Value> for CommandTool {
    fn declare(&self) -> crate::llm::function::FunctionDeclaration<CommandArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "command_tool",
            description: "Use this tool to execute OS commands with full interactive support, including handling of password prompts, confirmations, and user input.",
            parameters: schema_for!(CommandArgs),
            executor: Arc::new(CommandTool {
                input_callback: self.input_callback.clone(),
                additional_patterns: self.additional_patterns.clone(),
            }),
        }
    }
}
impl FnDeclarator<SignalArgs, serde_json::Value> for SignalTool {
    fn declare(&self) -> FunctionDeclaration<SignalArgs, serde_json::Value> {
        FunctionDeclaration {
            name: "signal_tool",
            description: "Use this tool to send signals to running processes by PID, such as terminating or interrupting them.",
            parameters: schema_for!(SignalArgs),
            executor: Arc::new(SignalTool {}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_command() {
        let tool = CommandTool::new();
        let args = CommandArgs {
            command: "echo".to_string(),
            args: vec!["hello".to_string()],
            working_dir: None,
            env: vec![],
            timeout_secs: 10,
            background: false,
            use_shell: false,
            custom_input_patterns: vec![],
        };

        let result = tool.execute_command(args).await.unwrap();
        assert_eq!(result.exit_code, Some(0));
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_command_with_shell() {
        let tool = CommandTool::new();
        let args = CommandArgs {
            command: "echo hello && echo world".to_string(),
            args: vec![],
            working_dir: None,
            env: vec![],
            timeout_secs: 10,
            background: false,
            use_shell: true,
            custom_input_patterns: vec![],
        };

        let result = tool.execute_command(args).await.unwrap();
        assert_eq!(result.exit_code, Some(0));
        assert!(result.stdout.contains("hello"));
        assert!(result.stdout.contains("world"));
    }

    #[tokio::test]
    async fn test_command_with_env() {
        let tool = CommandTool::new();
        let env = vec![EnvVar {
            name: "MY_VAR".to_string(),
            value: "my_value".to_string(),
        }];

        let args = CommandArgs {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), "echo $MY_VAR".to_string()],
            working_dir: None,
            env,
            timeout_secs: 10,
            background: false,
            use_shell: false,
            custom_input_patterns: vec![],
        };

        let result = tool.execute_command(args).await.unwrap();
        assert_eq!(result.exit_code, Some(0));
        assert!(result.stdout.contains("my_value"));
    }

    #[tokio::test]
    async fn test_command_timeout() {
        let tool = CommandTool::new();
        let args = CommandArgs {
            command: "sleep".to_string(),
            args: vec!["10".to_string()],
            working_dir: None,
            env: vec![],
            timeout_secs: 1,
            background: false,
            use_shell: false,
            custom_input_patterns: vec![],
        };

        let result = tool.execute_command(args).await.unwrap();
        assert!(result.timed_out);
    }

    #[tokio::test]
    async fn test_command_failure() {
        let tool = CommandTool::new();
        let args = CommandArgs {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), "exit 1".to_string()],
            working_dir: None,
            env: vec![],
            timeout_secs: 10,
            background: false,
            use_shell: false,
            custom_input_patterns: vec![],
        };

        let result = tool.execute_command(args).await.unwrap();
        assert_eq!(result.exit_code, Some(1));
    }

    #[test]
    fn test_input_detection() {
        let tool = CommandTool::new();

        assert!(tool.needs_input("Password:", &[]));
        assert!(tool.needs_input("Continue? [y/n]", &[]));
        assert!(tool.needs_input("[sudo] password for user:", &[]));
        assert!(tool.needs_input("Are you sure you want to continue?", &[]));
        assert!(!tool.needs_input("Processing files...", &[]));

        // Custom patterns
        assert!(tool.needs_input("custom prompt here", &["custom prompt".to_string()]));
    }

    #[test]
    fn test_default_timeout() {
        assert_eq!(default_timeout(), 300);
    }
}
