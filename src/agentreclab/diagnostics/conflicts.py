from agentreclab.state.snapshot import digest


class IntentMemoryConflictAnalyzer:
    """Opt-in lexical diagnostics; explicit constraints only, no baseline mutation."""

    def analyze(self, case_id, instruction, memories, forbidden_terms):
        findings = []
        for source_id, text in memories.items():
            for term in forbidden_terms:
                position = text.casefold().find(term.casefold())
                if term and position >= 0:
                    findings.append(
                        {
                            "case_id": case_id,
                            "label": "lexical_conflict_candidate",
                            "requires_human_review": True,
                            "constraint": term,
                            "instruction_sha256": digest(instruction),
                            "source_id": source_id,
                            "memory_sha256": digest(text),
                            "evidence": text[max(0, position - 40) : position + len(term) + 40],
                        }
                    )
        return findings
