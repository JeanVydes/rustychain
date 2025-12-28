use crate::execution::node::NodeId;
use crate::{FunctionCall, FunctionResult, Inference};

/// Events emitted during agent execution
#[derive(Debug, Clone)]
pub enum Event {
    // Observable events - notification only
    Started {
        message: String,
    },
    Thinking {
        node_id: NodeId,
    },
    StepCompleted {
        step: usize,
        node_id: NodeId,
    },
    Finished {
        inference: Inference,
    },
    Error {
        error: String,
    },

    // Interceptable events - can block and wait for response
    ToolCallRequested {
        node_id: NodeId,
        call: FunctionCall,
    },

    ToolCallCompleted {
        node_id: NodeId,
        call: FunctionCall,
        result: FunctionResult,
    },
}

/// Response from an interceptor listener
#[derive(Debug, Clone)]
pub enum InterceptionResponse {
    Continue,
    Modify { call: FunctionCall },
    Block { reason: String },
    Replace { result: FunctionResult },
}
