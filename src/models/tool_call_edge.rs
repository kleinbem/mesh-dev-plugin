//! Deterministic edge-case payloads for `dev/tool-call-edge`.
//!
//! Exercises split multi-chunk `<think>` tags containing decoy `</tool_call>` markers
//! and tool-call JSON arguments containing literal `<think>`, `</think>`, `<tool_call>`,
//! and `</tool_call>` tokens split mid-tag across SSE chunks.

use serde_json::{Value, json};

pub const TOOL_CALL_ID: &str = "call_mesh_dev_edge_0";
pub const TOOL_NAME: &str = "inspect_payload";

/// Reasoning fragments split intentionally mid-tag across multiple chunks,
/// including a decoy `</tool_call>` inside `<think>...</think>`.
pub const THINK_CHUNKS: &[&str] = &[
    "<th",
    "ink>planning tool invocation with literal </to",
    "ol_call> decoy inside reasoning</",
    "think>",
];

/// Tool-call JSON argument fragments split mid-tag and mid-escape across chunks.
/// Concatenated value is strictly valid JSON.
pub const TOOL_ARG_CHUNKS: &[&str] = &[
    r#"{"note":"contains <th"#,
    r#"ink>inner</th"#,
    r#"ink> and <tool_"#,
    r#"call>nested</to"#,
    r#"ol_call>","escaped_quote":"a\"</to"#,
    r#"ol_call>\"b","ok":true}"#,
];

/// Raw-tag mode chunks (`mode=raw-tag`) emitting raw `<tool_call>...</tool_call>` in `content`.
pub fn raw_tag_chunks() -> Vec<String> {
    let mut out: Vec<String> = THINK_CHUNKS.iter().map(|s| (*s).to_string()).collect();
    out.push("<tool_".to_string());
    out.push(format!("call>{{\"name\":\"{TOOL_NAME}\",\"arguments\":"));
    for frag in TOOL_ARG_CHUNKS {
        out.push((*frag).to_string());
    }
    out.push("}</to".to_string());
    out.push("ol_call>".to_string());
    out
}

pub fn full_think_content() -> String {
    THINK_CHUNKS.concat()
}

pub fn full_tool_arguments() -> String {
    TOOL_ARG_CHUNKS.concat()
}

pub fn non_streaming_tool_calls() -> Value {
    json!([
        {
            "id": TOOL_CALL_ID,
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "arguments": full_tool_arguments(),
            }
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenated_tool_arguments_are_valid_json_with_embedded_tags() {
        let raw = full_tool_arguments();
        let parsed: Value = serde_json::from_str(&raw).expect("valid JSON tool arguments");
        let note = parsed["note"].as_str().expect("note string");
        assert!(note.contains("<think>inner</think>"));
        assert!(note.contains("<tool_call>nested</tool_call>"));
        assert_eq!(
            parsed["escaped_quote"].as_str(),
            Some(r#"a"</tool_call>"b"#)
        );
        assert_eq!(parsed["ok"].as_bool(), Some(true));
    }

    #[test]
    fn raw_tag_chunks_contain_split_think_and_tool_call() {
        let joined = raw_tag_chunks().concat();
        assert!(joined.starts_with("<think>"));
        assert!(joined.contains("</think><tool_call>"));
        assert!(joined.ends_with("</tool_call>"));
    }
}
