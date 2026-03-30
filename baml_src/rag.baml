class ResponseWithCitations {
  answer string
  citations string[]
}

function AnswerWithCitations(question: string, context: string) -> ResponseWithCitations {
  client GraniteClient
  prompt #"
    You are a running biomechanics research assistant helping runners
    make informed decisions about their footwear and training.

    Guidelines for your answer:
    - Be practical and nuanced. Avoid extreme recommendations.
    - Focus on gradual transitions and individual differences.
    - Cite sources inline using their numbers, e.g. [1], [2].
    - Use only information from the provided context.
    - If the context doesn't contain an answer, say so clearly.

    Guidelines for citations:
    - Each entry in the citations array must follow this exact format:
      "[N] Paper Title (Year) — exact supporting sentence from context"
    - Only include citations that directly support a claim in your answer.

    QUESTION: {{ question }}

    RESEARCH CONTEXT:
    {{ context }}

    {{ ctx.output_format }}
  "#
}

test TestAnswerWithCitations {
  functions [AnswerWithCitations]
  args {
    question "Does heel-to-toe drop affect knee pain?"
    context #"
      [1] Masen Zhang, Huijuan Shi, Hui Liu and Xinglong Zhou (2021) | "Biomechanical Analysis of Running in Shoes with Different Heel-to-Toe Drops" | Section: conclusion
      Running in shoes with large drops significantly increased the knee extension moment and decreased the ankle eversion moment during the standing phase. These findings suggest that running shoes with large drops may be disadvantageous for runners with knee weakness and advantageous for runners with ankle weakness.

      ---

      [2] Masen Zhang, Huijuan Shi, Hui Liu and Xinglong Zhou (2021) | "Biomechanical Analysis of Running in Shoes with Different Heel-to-Toe Drops" | Section: discussion
      Running in shoes with large drops increased the peak knee flexion angle and extension moment, and decreased the peak ankle eversion angle and moment during the standing phase. Thus, the heel-to-toe drop of the running shoes was a key factor influencing the running pattern and the loading of the lower extremity joints.

      ---

      [3] Masen Zhang, Huijuan Shi, Hui Liu and Xinglong Zhou (2021) | "Biomechanical Analysis of Running in Shoes with Different Heel-to-Toe Drops" | Section: discussion
      Running in shoes with a 10-mm drop was associated with a higher rate of knee joint injuries (26%) compared with running in shoes without a drop (15%). These results indicate that wearing shoes with a 10-mm drop may be disadvantageous for runners with knee weakness and advantageous for runners with ankle weakness.
    "#
  }
}