use crate::config::Config;
use crate::events::LlmAction;
use anyhow::{Context, Result};
use tracing::{debug, warn};

/// LLM client for intent classification.
///
/// Sends user transcripts to an OpenAI-compatible chat completions API
/// and receives structured JSON responses classifying the intent as
/// dictation, command, or conversation.
pub struct LlmClient {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl LlmClient {
    /// Create a new LLM client with the given configuration.
    pub fn new(config: &Config) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        tracing::info!(
            "LLM client initialized: {} (model: {})",
            config.llm_base_url,
            config.llm_model
        );

        Self {
            client,
            base_url: config.llm_base_url.trim_end_matches('/').to_string(),
            api_key: config.llm_api_key.clone(),
            model: config.llm_model.clone(),
        }
    }

    /// Classify a user utterance by sending it to the LLM.
    ///
    /// The LLM returns a structured JSON response indicating whether the utterance
    /// is dictation (content to type), a command (editing instruction), or
    /// conversation (not to be typed).
    pub async fn classify_intent(
        &self,
        transcript: &str,
        document_context: &str,
    ) -> Result<LlmAction> {
        let system_prompt = build_system_prompt(document_context);

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            "response_format": { "type": "json_object" },
            "temperature": 0.1
        });

        debug!("Sending to LLM: \"{}\"", transcript);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to LLM API")?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".into());
            anyhow::bail!("LLM API returned error {}: {}", status, error_body);
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse LLM response as JSON")?;

        // Extract the message content from the OpenAI response format
        let content = response_json
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .context("LLM response missing choices[0].message.content")?;

        debug!("LLM raw response: {}", content);

        // Parse the JSON content into an LlmAction
        parse_llm_response(content)
    }
}

/// Parse the LLM's JSON response into an LlmAction.
///
/// Handles various response formats and falls back gracefully.
fn parse_llm_response(content: &str) -> Result<LlmAction> {
    // Try to parse directly as LlmAction
    match serde_json::from_str::<LlmAction>(content) {
        Ok(action) => {
            debug!("Parsed LLM action: {:?}", action);
            return Ok(action);
        }
        Err(e) => {
            debug!("Direct parse failed ({}), trying fallback", e);
        }
    }

    // Fallback: try to extract from nested JSON
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
        let intent = json
            .get("intent")
            .and_then(|v| v.as_str())
            .unwrap_or("dictate");

        match intent {
            "dictate" => {
                let text = json
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                return Ok(LlmAction::Dictate { text });
            }
            "command" => {
                let action_str = json
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("new_paragraph");

                let action = serde_json::from_value(serde_json::Value::String(
                    action_str.to_string(),
                ))
                .unwrap_or(crate::events::EditCommand::NewParagraph);

                let text = json
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                return Ok(LlmAction::Command { action, text });
            }
            "conversation" => {
                let response = json
                    .get("response")
                    .and_then(|v| v.as_str())
                    .unwrap_or("I understand.")
                    .to_string();
                return Ok(LlmAction::Conversation { response });
            }
            _ => {
                warn!("Unknown intent: {}", intent);
            }
        }
    }

    // Last resort: treat the whole thing as dictation
    warn!("Could not parse LLM response, treating as dictation: {}", content);
    Ok(LlmAction::Dictate {
        text: content.trim().to_string(),
    })
}

/// Build the system prompt for the LLM intent classifier.
fn build_system_prompt(document_context: &str) -> String {
    format!(
        r#"You are a professional typist AI assistant. The user is dictating a document to you through voice input. Your job is to intelligently classify each spoken utterance and respond with a JSON object.

CONTEXT:
- The user speaks naturally. They may dictate content, give editing commands, or have casual conversation.
- You must accurately distinguish between these three intents.
- For dictation, clean up the speech: remove filler words (um, uh, like, you know, basically, 那个, 就是, 嗯), fix grammar, add proper punctuation, and produce clean written text.
- The user may speak in English or Mandarin (中文). Handle both languages naturally. Output in whatever language the user speaks.

CURRENT DOCUMENT STATE:
{document_context}

RESPONSE FORMAT — Respond with ONLY a JSON object (no other text, no markdown):

1. DICTATION — The user is saying content to be typed into the document:
   {{"intent": "dictate", "text": "The cleaned, properly formatted text."}}

   Examples:
   - "um so the project deadline is uh next Friday I think" → {{"intent": "dictate", "text": "The project deadline is next Friday."}}
   - "第一点我们需要完成市场调研" → {{"intent": "dictate", "text": "第一点，我们需要完成市场调研。"}}
   - "dear Mr Johnson comma I am writing to inform you period" → {{"intent": "dictate", "text": "Dear Mr. Johnson, I am writing to inform you."}}
   - "write down meeting notes for today" followed by "first item review the budget" → {{"intent": "dictate", "text": "First item: review the budget."}}

2. COMMAND — The user is giving an editing instruction:
   {{"intent": "command", "action": "<action_name>", "text": "<optional text>"}}

   Available actions:
   - "new_paragraph" — Start a new paragraph
   - "delete_last_sentence" — Delete the last sentence
   - "delete_last_paragraph" — Delete the last paragraph
   - "undo" — Undo the last change
   - "redo" — Redo the last undone change
   - "clear_all" — Clear the entire document
   - "insert_heading" — Insert a heading (requires "text" field)
   - "insert_bullet" — Insert a bullet point (requires "text" field)

   Examples:
   - "new paragraph" → {{"intent": "command", "action": "new_paragraph"}}
   - "delete that last sentence" / "删掉最后一句" → {{"intent": "command", "action": "delete_last_sentence"}}
   - "add a heading that says Introduction" → {{"intent": "command", "action": "insert_heading", "text": "Introduction"}}
   - "undo" / "撤销" → {{"intent": "command", "action": "undo"}}

3. CONVERSATION — The user is talking to you, not dictating:
   {{"intent": "conversation", "response": "Your brief reply."}}

   Examples:
   - "what have I written so far?" / "我写了什么" → {{"intent": "conversation", "response": "You've written ..."}}
   - "hmm let me think" → {{"intent": "conversation", "response": "Take your time."}}
   - "how many words is that?" → {{"intent": "conversation", "response": "Your document has X words."}}

RULES:
- Respond with ONLY the JSON object.
- When ambiguous, lean toward dictation — the user's primary goal is typing.
- Phrases like "write down", "type", "put", "add" indicate dictation.
- Phrases like "delete", "remove", "undo", "new paragraph", "heading", "撤销", "删除" indicate commands.
- Questions, thinking aloud, and meta-commentary indicate conversation."#,
        document_context = document_context
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dictate() {
        let input = r#"{"intent": "dictate", "text": "Hello world."}"#;
        let action = parse_llm_response(input).unwrap();
        match action {
            LlmAction::Dictate { text } => assert_eq!(text, "Hello world."),
            _ => panic!("Expected Dictate"),
        }
    }

    #[test]
    fn test_parse_command() {
        let input = r#"{"intent": "command", "action": "new_paragraph"}"#;
        let action = parse_llm_response(input).unwrap();
        match action {
            LlmAction::Command { action, text } => {
                assert_eq!(action, crate::events::EditCommand::NewParagraph);
                assert!(text.is_none());
            }
            _ => panic!("Expected Command"),
        }
    }

    #[test]
    fn test_parse_conversation() {
        let input = r#"{"intent": "conversation", "response": "Take your time."}"#;
        let action = parse_llm_response(input).unwrap();
        match action {
            LlmAction::Conversation { response } => assert_eq!(response, "Take your time."),
            _ => panic!("Expected Conversation"),
        }
    }

    #[test]
    fn test_parse_command_with_text() {
        let input = r#"{"intent": "command", "action": "insert_heading", "text": "Introduction"}"#;
        let action = parse_llm_response(input).unwrap();
        match action {
            LlmAction::Command { action, text } => {
                assert_eq!(action, crate::events::EditCommand::InsertHeading);
                assert_eq!(text, Some("Introduction".to_string()));
            }
            _ => panic!("Expected Command with text"),
        }
    }
}
