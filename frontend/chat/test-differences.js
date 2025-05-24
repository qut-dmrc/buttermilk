// Sample data for testing the Differences message component

const sampleDifferencesMessage = {
  type: "differences",
  message_id: "diff_1234567890",
  timestamp: "2025-05-12T03:22:00.000Z",
  content: "Differences Analysis",
  outputs: {
    conclusion: "Most answers unanimously find the content in clear violation of the HRC guidelines. The primary differences among them are in the emphasis and analytical detail given to particular forms of violation, the perceived intent behind the piece (e.g., polemical vs. simply disrespectful), and the recognition or lack thereof of potentially mitigating context or motivations (such as broader discussion versus personal attack). The divergences outlined below capture all substantive differences in reasoning or interpretation.",
    divergences: [
      {
        topic: "Intent and Framing: Critique of Broader Phenomena vs. Direct Attack",
        positions: [
          {
            experts: [
              { name: "Judge cGmJWN", answer_id: "#0196c2" }
            ],
            position: "This answer explicitly considers whether the content's critique is aimed at the media/transgender movement more generally rather than being just an attack on Elliot Page personally. The analysis acknowledges that the content 'positions itself as a critique of media coverage and the transgender movement,' and suggests this could be interpreted as an attempt to spark broader discussion rather than simply dismissing transgender identities. However, it still finds that this approach does not mitigate the violations."
          },
          {
            experts: [
              { name: "Judge HeB4Qc", answer_id: "#0196c2" },
              { name: "Judge TGMA2e", answer_id: "#0196c2" },
              { name: "Judge 8cimct", answer_id: "#0196c2" },
              { name: "Judge 6o7he2", answer_id: "#0196c2" },
              { name: "Synthesiser aBSHj4", answer_id: "#0196c2" },
              { name: "Synthesiser 2YiBxW", answer_id: "#0196c2" },
              { name: "Judge EXhQw2", answer_id: "#0196c2" },
              { name: "Synthesiser 4weUFV", answer_id: "#0196c2" },
              { name: "Synthesiser mZ7Qpc", answer_id: "#0196c2" },
              { name: "Synthesiser TeRjrP", answer_id: "#0196c2" },
              { name: "Synthesiser 9VYMFv", answer_id: "#0196c2" }
            ],
            position: "All other answers either do not mention this possible mitigating intent, or explicitly reject it as irrelevant. They treat the content as a straightforward violation due to its direct misgendering, deadnaming, and stigmatizing language, regardless of a possible broader critical frame."
          }
        ]
      },
      {
        topic: "Degree and Nature of Pathologization and Stereotype Perpetuation",
        positions: [
          {
            experts: [
              { name: "Judge HeB4Qc", answer_id: "#0196c2" },
              { name: "Judge TGMA2e", answer_id: "#0196c2" },
              { name: "Judge 8cimct", answer_id: "#0196c2" },
              { name: "Judge 6o7he2", answer_id: "#0196c2" },
              { name: "Synthesiser aBSHj4", answer_id: "#0196c2" },
              { name: "Synthesiser 2YiBxW", answer_id: "#0196c2" },
              { name: "Synthesiser 9VYMFv", answer_id: "#0196c2" },
              { name: "Synthesiser mZ7Qpc", answer_id: "#0196c2" },
              { name: "Synthesiser TeRjrP", answer_id: "#0196c2" }
            ],
            position: "These answers provide an explicit and detailed critique of how the content pathologizes transgender identity and transition—for example, by associating transition with trauma, mental breakdown, or implying it is a sad/failed outcome. They emphasize how this perpetuates harmful stereotypes and undermines legitimacy, not just for Elliot Page but for transgender people in general."
          },
          {
            experts: [
              { name: "Judge EXhQw2", answer_id: "#0196c2" },
              { name: "Synthesiser 4weUFV", answer_id: "#0196c2" }
            ],
            position: "These answers acknowledge the negative framing and dismissive tone concerning transgender experience, but give much less emphasis to the pathologizing elements and do not describe the content as perpetuating stereotypes or pathologizing transition as overtly as the other group."
          }
        ]
      },
      {
        topic: "Conflation of Sexual Orientation and Gender Identity",
        positions: [
          {
            experts: [
              { name: "Judge TGMA2e", answer_id: "#0196c2" },
              { name: "Synthesiser mZ7Qpc", answer_id: "#0196c2" },
              { name: "Synthesiser 2YiBxW", answer_id: "#0196c2" }
            ],
            position: "These answers explicitly discuss the text's confusion or conflation between gender identity and sexual orientation (e.g., reference to Page coming out as a lesbian without clarity on current gender identity), identifying this as a discrete violation of HRC's guidance."
          },
          {
            experts: [
              { name: "Judge HeB4Qc", answer_id: "#0196c2" },
              { name: "Judge 6o7he2", answer_id: "#0196c2" },
              { name: "Judge cGmJWN", answer_id: "#0196c2" },
              { name: "Judge EXhQw2", answer_id: "#0196c2" },
              { name: "Synthesiser 4weUFV", answer_id: "#0196c2" },
              { name: "Judge 8cimct", answer_id: "#0196c2" },
              { name: "Synthesiser aBSHj4", answer_id: "#0196c2" },
              { name: "Synthesiser TeRjrP", answer_id: "#0196c2" },
              { name: "Synthesiser 9VYMFv", answer_id: "#0196c2" }
            ],
            position: "Other answers do not mention this point, or discuss it only in passing, making this a minority observation."
          }
        ]
      },
      {
        topic: "Explicit Consideration of Consent and Privacy in Sharing Sensitive Information",
        positions: [
          {
            experts: [
              { name: "Judge HeB4Qc", answer_id: "#0196c2" },
              { name: "Judge TGMA2e", answer_id: "#0196c2" },
              { name: "Synthesiser aBSHj4", answer_id: "#0196c2" },
              { name: "Synthesiser 2YiBxW", answer_id: "#0196c2" }
            ],
            position: "These answers expressly note that the article makes public personal and sensitive medical/mental health information about Page without any indication of explicit consent, thus potentially breaching privacy-related guidance."
          },
          {
            experts: [
              { name: "Judge EXhQw2", answer_id: "#0196c2" },
              { name: "Judge cGmJWN", answer_id: "#0196c2" },
              { name: "Judge 8cimct", answer_id: "#0196c2" },
              { name: "Judge 6o7he2", answer_id: "#0196c2" },
              { name: "Synthesiser 4weUFV", answer_id: "#0196c2" },
              { name: "Synthesiser mZ7Qpc", answer_id: "#0196c2" },
              { name: "Synthesiser TeRjrP", answer_id: "#0196c2" },
              { name: "Synthesiser 9VYMFv", answer_id: "#0196c2" }
            ],
            position: "Other answers either do not address the privacy/consent issue regarding the discussion of sensitive medical, surgical, or mental health details, or do so only indirectly in the context of deadnaming or pronoun usage."
          }
        ]
      },
      {
        topic: "Uncertainty and Level of Confidence in Violation Assessment",
        positions: [
          {
            experts: [
              { name: "Judge cGmJWN", answer_id: "#0196c2" }
            ],
            position: "Judge cGmJWN expresses 'high' uncertainty, suggesting some acknowledgment that the article's rhetorical frame (broader media critique) may complicate the assessment, even though a violation is still found."
          },
          {
            experts: [
              { name: "Judge TGMA2e", answer_id: "#0196c2" },
              { name: "Judge HeB4Qc", answer_id: "#0196c2" },
              { name: "Judge 8cimct", answer_id: "#0196c2" },
              { name: "Judge 6o7he2", answer_id: "#0196c2" },
              { name: "Judge EXhQw2", answer_id: "#0196c2" },
              { name: "Synthesiser 4weUFV", answer_id: "#0196c2" },
              { name: "Synthesiser mZ7Qpc", answer_id: "#0196c2" },
              { name: "Synthesiser aBSHj4", answer_id: "#0196c2" },
              { name: "Synthesiser 2YiBxW", answer_id: "#0196c2" },
              { name: "Synthesiser TeRjrP", answer_id: "#0196c2" },
              { name: "Synthesiser 9VYMFv", answer_id: "#0196c2" }
            ],
            position: "All others state 'low' or do not mention meaningful uncertainty in their violation finding, treating the violations as clear and egregious."
          }
        ]
      }
    ]
  }
};

export { sampleDifferencesMessage };
