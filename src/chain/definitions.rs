use std::{any::Any, marker::PhantomData, sync::Arc};

use futures_core::future::BoxFuture;

use crate::chain::step::{Runnable, RunnableWrapper};

pub type IO = Arc<dyn Any + Send + Sync>;

#[derive(Clone)]
pub struct Chain<I, O> {
    steps: Vec<(String, RunnableWrapper)>,
    current_index: usize,
    current_value: Option<IO>,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O> Chain<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            current_index: 0,
            current_value: None,
            _phantom: PhantomData,
        }
    }

    pub fn with_input(input: I) -> Self {
        Self {
            steps: Vec::new(),
            current_index: 0,
            current_value: Some(Arc::new(input)),
            _phantom: PhantomData,
        }
    }

    pub fn add_step<R, I2, O2>(mut self, name: impl Into<String>, runnable: R) -> Self
    where
        R: Runnable<I2, O2> + 'static,
        I2: Send + Sync + 'static + Clone,
        O2: Send + Sync + 'static,
    {
        let runnable = Arc::new(runnable);
        let name = name.into();

        self.steps.push((
            name,
            Arc::new(move |input: Arc<dyn Any + Send + Sync>| {
                let r = runnable.clone();
                Box::pin(async move {
                    let typed_input = input
                        .downcast::<I2>()
                        .map_err(|err| crate::CoreError::Downcast(err))?;

                    let t = typed_input.clone();
                    let result = r.run(t).await?;
                    Ok(Arc::new(result) as Arc<dyn Any + Send + Sync>)
                })
            }),
        ));

        self
    }

    /// Set input and returns the chain
    pub fn start(mut self, input: I) -> Self {
        self.current_value = Some(Arc::new(input));
        self.current_index = 0;
        self
    }

    /// Executes the next step in the chain
    pub async fn next(&mut self) -> Option<crate::Result<StepResult>> {
        if self.current_index >= self.steps.len() {
            log::debug!("No more steps to execute in the chain.");
            return None;
        }

        log::trace!(
            "Preparing to execute step {} of {}.",
            self.current_index + 1,
            self.steps.len()
        );

        let current = self.current_value.take()?;
        let (name, step) = &self.steps[self.current_index];
        let input = current.clone();

        log::trace!(
            "[{} ({}/{})] ",
            name,
            self.current_index + 1,
            self.steps.len()
        );

        match step(current).await {
            Ok(result) => {
                let step_result = StepResult {
                    index: self.current_index,
                    name: name.clone(),
                    input,
                    output: result.clone(),
                };

                log::trace!(
                    "[{} ({}/{})] Step executed successfully. \n\n{}\n{}\n{}",
                    name,
                    self.current_index + 1,
                    self.steps.len(),
                    "----------------------------------------",
                    format!("{:?}", step_result.output),
                    "----------------------------------------"
                );

                self.current_index += 1;
                self.current_value = Some(result);

                Some(Ok(step_result))
            }
            Err(e) => {
                log::error!(
                    "[{} ({}/{})] Step execution failed: {}",
                    name,
                    self.current_index + 1,
                    self.steps.len(),
                    e
                );

                self.current_value = None;
                Some(Err(e))
            }
        }
    }

    /// Executes all remaining steps in the chain
    pub async fn run_remaining(&mut self) -> crate::Result<()> {
        log::trace!(
            "Running remaining steps from index {} of {}.",
            self.current_index + 1,
            self.steps.len()
        );

        while let Some(result) = self.next().await {
            result?;
        }
        Ok(())
    }

    /// Executes the entire chain from the beginning with the given input and returns the final output
    pub async fn run(mut self, input: I) -> crate::Result<Arc<O>> {
        log::trace!("Running the entire chain from the beginning.");

        self.current_value = Some(Arc::new(input));
        self.current_index = 0;

        for (index, (name, step)) in self.steps.iter().enumerate() {
            log::trace!(
                "[{} ({}/{})] Executing step...",
                name,
                index + 1,
                self.steps.len()
            );

            let current = self
                .current_value
                .take()
                .ok_or_else(|| crate::CoreError::StepError {
                    index,
                    step_name: Some(name.clone()),
                    source: Box::new(crate::CoreError::NotInput),
                })?;

            log::trace!("({}/{}) Step input prepared.", index + 1, self.steps.len());

            self.current_value = Some(step(current).await?);

            log::trace!(
                "[{} ({}/{})] Step executed successfully.",
                name,
                index + 1,
                self.steps.len()
            );

            self.current_index = index + 1;
        }

        log::trace!("All steps executed. Finalizing chain.");

        self.finalize()
    }

    /// Executes the entire chain from the beginning with the given input and an interceptor function
    pub async fn run_with_interceptor<F>(
        mut self,
        input: I,
        mut interceptor: F,
    ) -> crate::Result<Arc<O>>
    where
        F: FnMut(usize, &str, &dyn std::any::Any) -> BoxFuture<'static, crate::Result<()>>,
    {
        log::trace!("Running the entire chain with interceptor from the beginning.");
        self.current_value = Some(Arc::new(input));
        self.current_index = 0;

        for (index, (name, step)) in self.steps.iter().enumerate() {
            log::trace!(
                "[{} ({}/{})] Executing step...",
                name,
                index + 1,
                self.steps.len()
            );

            let current = self
                .current_value
                .take()
                .ok_or_else(|| crate::CoreError::StepError {
                    index,
                    step_name: Some(name.clone()),
                    source: Box::new(crate::CoreError::NotInput),
                })?;

            log::trace!(
                "[{} ({}/{})] Step input prepared.",
                name,
                index + 1,
                self.steps.len()
            );

            let result = step(current).await?;

            log::trace!(
                "[{} ({}/{})] Step executed successfully.",
                name,
                index + 1,
                self.steps.len()
            );

            log::trace!(
                "[{} ({}/{})] Executing interceptor...",
                name,
                index + 1,
                self.steps.len()
            );

            interceptor(index, name, result.as_ref()).await?;

            log::trace!(
                "[{} ({}/{})] Interceptor executed successfully.",
                name,
                index + 1,
                self.steps.len()
            );

            self.current_value = Some(result);
            self.current_index = index + 1;
        }

        self.finalize()
    }

    /// Finalizes the chain and returns the final output
    pub fn finalize(self) -> crate::Result<Arc<O>> {
        log::trace!("Finalizing the chain.");

        if self.current_index != self.steps.len() {
            return Err(Box::from(crate::CoreError::ChainNotFinalized));
        }

        let final_value = self
            .current_value
            .ok_or_else(|| crate::CoreError::StepError {
                index: self.current_index,
                step_name: None,
                source: Box::new(crate::CoreError::StepError {
                    index: self.current_index,
                    step_name: None,
                    source: Box::new(crate::CoreError::NotInput),
                }),
            })?;

        let output = final_value
            .downcast::<O>()
            .map_err(|err| crate::CoreError::Downcast(err))?;

        Ok(output)
    }

    /// Get current value as Any
    pub fn current_value(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.current_value.clone()
    }

    /// Resets the chain to the initial state with the given input
    pub fn reset(mut self, input: I) -> Self {
        self.current_index = 0;
        self.current_value = Some(Arc::new(input));
        self
    }

    /// Total number of steps in the chain
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Current step index
    pub fn current_step(&self) -> usize {
        self.current_index
    }

    /// Checks if there are more steps to execute
    pub fn has_next(&self) -> bool {
        self.current_index < self.steps.len()
    }
}

pub struct StepResult {
    pub index: usize,
    pub name: String,
    pub input: Arc<dyn std::any::Any + Send + Sync>,
    pub output: Arc<dyn std::any::Any + Send + Sync>,
}

impl StepResult {
    /// Helper para hacer downcast del valor a un tipo concreto
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.output.downcast_ref::<T>()
    }

    /// Consume el resultado y hace downcast
    pub fn downcast<T: Send + Sync + 'static>(self) -> Option<Arc<T>> {
        self.output.downcast::<T>().ok()
    }
}
