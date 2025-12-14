use std::{any::Any, sync::Arc};

use futures_core::future::BoxFuture;

#[async_trait::async_trait]
pub trait Runnable<I, O>: Send + Sync
where
    I: Send + Sync,
    O: Send + Sync,
{
    async fn run(&self, input: Arc<I>) -> crate::Result<O>;
}

pub type RunnableWrapper = Arc<
    dyn Fn(
            Arc<dyn Any + Send + Sync>,
        ) -> BoxFuture<'static, crate::Result<Arc<dyn Any + Send + Sync>>>
        + Send
        + Sync,
>;

#[macro_export]
macro_rules! impl_runnable {
    ($type:ty, $input:ty, $output:ty) => {
        impl Runnable<$input, $output> for $type {
            fn run(&self, input: $input) -> BoxFuture<'_, $crate::Result<$output>> {
                Box::pin(async move { self.run_impl(input).await })
            }
        }
    };
}