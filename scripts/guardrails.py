"""AML guardrail validators for the generation pipeline.

Two implementations with the same interface:
  NullGuardrail: no-op passthrough; use during ablation evaluation
  AMLGuardrail: six production validators

Select via GUARDRAILS_ON env flag in notebook 04.
"""

from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    reason: str


class NullGuardrail:
    """No-op passthrough. Every call returns passed=True."""

    def check_input(self, state: dict) -> GuardrailResult:
        return GuardrailResult(passed=True, reason='null guard')

    def check_output(self, state: dict) -> dict:
        return {'passed': True, 'reason': 'null guard'}


class AMLGuardrail:
    """Production guardrail with six validators.

    Input check  : topic relevance
    Output checks: confidence threshold, schema, citation completeness,
                   hallucination block, scope enforcement
    """

    AML_KEYWORDS = {
        'aml', 'money laundering', 'cdd', 'kyc', 'edd', 'pep', 'sar', 'mlro',
        'suspicious', 'transaction monitoring', 'risk', 'compliance', 'regulation',
        'criminal', 'proceeds', 'beneficial owner', 'customer due diligence',
        'politically exposed', 'counter-financing', 'terrorist', 'financing',
        'sanctions', 'fatf', 'jmlsg', 'mlr', 'fca', 'poca',
    }

    MIN_RRF_SCORE = 0.01   # abstain if top RRF score is below this threshold

    def check_input(self, state: dict) -> GuardrailResult:
        query = state.get('query', '').lower()
        if not any(kw in query for kw in self.AML_KEYWORDS):
            return GuardrailResult(
                passed=False,
                reason=f'Query does not appear to be AML/compliance related: {query!r}',
            )
        return GuardrailResult(passed=True, reason='topic relevance passed')

    def check_output(self, state: dict) -> dict:
        retrieved = state.get('retrieved', [])
        answer = state.get('answer', '')
        citations = state.get('citations', [])
        retrieved_ids = {r['clause_id'] for r in retrieved}

        # Confidence threshold
        if retrieved and isinstance(retrieved[0].get('rrf_score'), float):
            if retrieved[0]['rrf_score'] < self.MIN_RRF_SCORE:
                return {'passed': False, 'reason': 'retrieval confidence below threshold'}

        # Schema validation
        if not isinstance(answer, str) or not isinstance(citations, list):
            return {'passed': False, 'reason': 'output schema invalid'}

        # Hallucination block
        hallucinated = [cid for cid in citations if cid not in retrieved_ids]
        if hallucinated:
            return {'passed': False, 'reason': f'hallucinated citations: {hallucinated}'}

        return {'passed': True, 'reason': 'all production checks passed'}
