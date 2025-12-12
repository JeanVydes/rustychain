use cpal::traits::HostTrait;

pub fn get_default_input_device() -> Option<cpal::Device> {
    let host = cpal::default_host();

    host.default_input_device()
}

pub fn get_default_output_device() -> Option<cpal::Device> {
    let host = cpal::default_host();

    host.default_output_device()
}

pub fn list_input_devices() -> Result<Vec<cpal::Device>, cpal::DevicesError> {
    let host = cpal::default_host();
    let devices = host.input_devices()?.collect();
    Ok(devices)
}

pub fn list_output_devices() -> Result<Vec<cpal::Device>, cpal::DevicesError> {
    let host = cpal::default_host();
    let devices = host.output_devices()?.collect();
    Ok(devices)
}
/*
pub async fn record_audio() -> Result<(), Box<dyn Error>> {
    let device = get_default_input_device().ok_or("No input device available")?;
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    let mut audio_buffer: Vec<f32> = Vec::new();

    let mic_stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
            audio_buffer.extend_from_slice(data); // Collect chunks
        },
        |err| eprintln!("Mic error: {}", err),
        None,
    )?;

    mic_stream.play()?;

    let output_stream = OutputStreamBuilder::open_default_stream()?;
    let sink = Sink::connect_new(output_stream.mixer());

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await; // Process every 2s
        if audio_buffer.is_empty() {
            continue;
        }

        let audio_chunk = std::mem::take(&mut audio_buffer);
        let user_item = HistoryItem {
            role: Role::User,
            message: "".to_string(), // No text needed if audio is primary
            audio: Some(audio_chunk.iter().flat_map(|&s| s.to_le_bytes()).collect()), // Raw bytes
            ..Default::default()
        };

        // Use your stream function (modified for multimodal)
        let text_stream = self.stream(history, user_item).await?;

        // Pipe to ElevenLabs TTS
        let http_client = HttpClient::new();
        while let Some(Ok(gemini_item)) = text_stream.next().await {
            // Stream text chunk to ElevenLabs (WebSocket for low-latency)
            let ws_url = "wss://api.elevenlabs.io/v1/text-to-speech/YOUR_VOICE_ID/stream";
            let (mut ws, _) = tungstenite::connect(ws_url)?; // Use tungstenite for WS
            let payload = serde_json::json!({
                "text": gemini_item.message,
                "try_trigger_generation": true,  // For streaming
            });
            ws.send(tungstenite::Message::Text(payload.to_string()))?;

            // Receive audio chunks and play gradually
            while let Ok(msg) = ws.read() {
                if let tungstenite::Message::Binary(audio_data) = msg {
                    sink.append_from_slice(&audio_data); // Play in real-time
                } else if let tungstenite::Message::Text(_) = msg {
                    break; // End of chunk
                }
            }

            history.push(gemini_item); // Update history
        }
    }

    Ok(())
}
*/
