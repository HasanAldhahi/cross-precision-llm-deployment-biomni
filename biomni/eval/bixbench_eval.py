"""
BixBenchEval: Evaluation loader for BixBench benchmark

This class provides a unified interface to evaluate user answers against ground truth
for all questions in the BixBench benchmark.
"""

import json
import re
from typing import Any, Optional
from pathlib import Path


class BixBenchEval:
    """
    Evaluation loader for BixBench benchmark

    Usage:
        evaluator = BixBenchEval('BixBench.jsonl')
        score = evaluator.evaluate('bix-1-q1', '0.0002')
    """

    def __init__(self, dataset_path: str = None):
        """
        Initialize the BixBenchEval evaluator

        Args:
            dataset_path: Path to the BixBench.jsonl file
        """
        if dataset_path is None:
            dataset_path = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env/biomni_data/benchmark/BixBench/BixBench.jsonl"
        
        self.dataset_path = dataset_path
        self.instances = []
        self.instance_map = {}  # question_id -> index
        self.id_map = {}  # id (uuid) -> index
        
        # Load JSONL file
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    instance = json.loads(line)
                    self.instances.append(instance)
                    # Index by question_id
                    question_id = instance.get('question_id', instance.get('id'))
                    self.instance_map[question_id] = idx
                    # Also index by UUID id
                    self.id_map[instance['id']] = idx

        print(f"Loaded BixBench dataset: {len(self.instances)} instances")

    def evaluate(self, question_id: str, user_answer: str) -> float:
        """
        Evaluate a user's answer for a given question

        Args:
            question_id: Question ID (e.g., 'bix-1-q1') or UUID
            user_answer: User's answer

        Returns:
            float: Reward score (0.0 to 1.0)
        """
        # Look up the instance
        if question_id in self.instance_map:
            idx = self.instance_map[question_id]
        elif question_id in self.id_map:
            idx = self.id_map[question_id]
        else:
            raise ValueError(f"Question not found: {question_id}")

        instance = self.instances[idx]
        ideal = instance['ideal']
        eval_mode = instance.get('eval_mode', 'str_verifier')

        try:
            reward = self._compute_reward(user_answer, ideal, eval_mode)
            return float(reward)
        except Exception as e:
            raise RuntimeError(f"Error computing reward for {question_id}: {e}")

    def _compute_reward(self, user_answer: str, ideal: str, eval_mode: str) -> float:
        """Compute reward using eval_mode specific logic"""
        
        if not user_answer or not isinstance(user_answer, str):
            return 0.0
        
        user_answer = user_answer.strip()
        ideal = str(ideal).strip()
        
        if eval_mode == 'str_verifier':
            return self._str_verify(user_answer, ideal)
        elif eval_mode == 'range_verifier':
            return self._range_verify(user_answer, ideal)
        elif eval_mode == 'llm_verifier':
            # For LLM verifier, we use fuzzy string matching as fallback
            # In production, this would call an LLM
            return self._fuzzy_verify(user_answer, ideal)
        else:
            # Default to string verification
            return self._str_verify(user_answer, ideal)

    def _str_verify(self, user_answer: str, ideal: str) -> float:
        """String verification - exact or normalized match"""
        # Normalize both answers
        user_norm = self._normalize_answer(user_answer)
        ideal_norm = self._normalize_answer(ideal)
        
        # Exact match after normalization
        if user_norm == ideal_norm:
            return 1.0
        
        # Check if ideal is contained in user answer (for longer responses)
        if ideal_norm in user_norm:
            return 1.0
        
        # Try extracting numbers if both are numeric
        user_num = self._extract_number(user_answer)
        ideal_num = self._extract_number(ideal)
        
        if user_num is not None and ideal_num is not None:
            # Allow for small floating point differences
            if abs(user_num - ideal_num) < 1e-6:
                return 1.0
            # Allow for percentage vs decimal (e.g., 35% vs 0.35)
            if abs(user_num - ideal_num * 100) < 1e-6 or abs(user_num * 100 - ideal_num) < 1e-6:
                return 1.0
        
        return 0.0

    def _range_verify(self, user_answer: str, ideal: str) -> float:
        """Range verification - check if answer falls within a range"""
        # Parse the range from ideal (format: "(low, high)" or "(low,high)")
        range_match = re.search(r'\(?\s*([\d.Ee\-+]+)\s*,\s*([\d.Ee\-+]+)\s*\)?', ideal)
        
        if not range_match:
            # If ideal is not a range, fall back to string verification
            return self._str_verify(user_answer, ideal)
        
        try:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
        except ValueError:
            return self._str_verify(user_answer, ideal)
        
        # Extract number from user answer
        user_num = self._extract_number(user_answer)
        
        if user_num is not None:
            if low <= user_num <= high:
                return 1.0
            # Check percentage format (e.g., 35% for range (0.35, 0.40))
            if low <= user_num / 100 <= high:
                return 1.0
        
        return 0.0

    def _fuzzy_verify(self, user_answer: str, ideal: str) -> float:
        """Fuzzy verification for llm_verifier mode"""
        # First try exact string match
        if self._str_verify(user_answer, ideal) == 1.0:
            return 1.0
        
        # Try to find the ideal answer within the user's response
        user_lower = user_answer.lower()
        ideal_lower = ideal.lower()
        
        # Check for key phrases
        if ideal_lower in user_lower:
            return 1.0
        
        # For numeric answers, try range-like matching
        user_num = self._extract_number(user_answer)
        ideal_num = self._extract_number(ideal)
        
        if user_num is not None and ideal_num is not None:
            # Allow 10% tolerance for fuzzy matching
            tolerance = abs(ideal_num * 0.1) if ideal_num != 0 else 0.1
            if abs(user_num - ideal_num) <= tolerance:
                return 1.0
        
        return 0.0

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison"""
        # Convert to lowercase
        answer = answer.lower().strip()
        # Remove common punctuation
        answer = re.sub(r'[,\s]+', ' ', answer)
        # Remove leading/trailing spaces
        answer = answer.strip()
        # Remove percentage sign and convert
        answer = answer.replace('%', '')
        return answer

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a number from text"""
        # Handle scientific notation
        sci_match = re.search(r'([\d.]+)[Ee]([\-+]?\d+)', text)
        if sci_match:
            try:
                return float(sci_match.group(0))
            except ValueError:
                pass
        
        # Handle percentage
        pct_match = re.search(r'([\d.]+)\s*%', text)
        if pct_match:
            try:
                return float(pct_match.group(1))
            except ValueError:
                pass
        
        # Handle regular numbers (with possible commas)
        num_match = re.search(r'[\d,]+\.?\d*', text.replace(',', ''))
        if num_match:
            try:
                return float(num_match.group(0).replace(',', ''))
            except ValueError:
                pass
        
        return None

    def get_instance(self, question_id: str) -> dict[str, Any]:
        """
        Get information about a specific instance

        Args:
            question_id: Question ID or UUID

        Returns:
            dict: Instance information
        """
        if question_id in self.instance_map:
            idx = self.instance_map[question_id]
        elif question_id in self.id_map:
            idx = self.id_map[question_id]
        else:
            raise ValueError(f"Question not found: {question_id}")

        return self.instances[idx].copy()

    def get_all_instances(self) -> list[dict]:
        """Get all instances as a list of dicts"""
        return [inst.copy() for inst in self.instances]

    def list_question_ids(self) -> list:
        """Get list of all question IDs"""
        return list(self.instance_map.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the dataset"""
        eval_modes = {}
        categories = {}
        
        for inst in self.instances:
            mode = inst.get('eval_mode', 'unknown')
            eval_modes[mode] = eval_modes.get(mode, 0) + 1
            
            cats = inst.get('categories', '')
            if isinstance(cats, str):
                for cat in cats.split(','):
                    cat = cat.strip()
                    if cat:
                        categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_instances': len(self.instances),
            'eval_modes': eval_modes,
            'categories': categories,
        }

    def batch_evaluate(self, evaluations: list) -> list:
        """
        Evaluate multiple instances at once

        Args:
            evaluations: List of tuples (question_id, user_answer)

        Returns:
            list: List of reward scores
        """
        results = []
        for question_id, user_answer in evaluations:
            try:
                score = self.evaluate(question_id, user_answer)
                results.append(score)
            except Exception as e:
                print(f"Error evaluating {question_id}: {e}")
                results.append(0.0)

        return results

    def __repr__(self):
        return f"BixBenchEval(instances={len(self.instances)})"

    def __len__(self):
        return len(self.instances)


def main():
    """Demo usage of BixBenchEval"""
    evaluator = BixBenchEval()

    print("\nDataset statistics:")
    stats = evaluator.get_stats()
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Eval modes: {stats['eval_modes']}")

    print("\nFirst 5 question IDs:")
    for qid in evaluator.list_question_ids()[:5]:
        print(f"  - {qid}")

    # Example evaluation
    print("\n" + "=" * 60)
    print("Example evaluation:")
    print("=" * 60)

    # Get first instance
    first_instance = evaluator.instances[0]
    question_id = first_instance['question_id']
    ideal = first_instance['ideal']
    eval_mode = first_instance['eval_mode']

    print(f"\nQuestion ID: {question_id}")
    print(f"Eval mode: {eval_mode}")
    print(f"Ideal answer: {ideal}")
    print(f"Question preview: {first_instance['question'][:200]}...")

    # Test with correct answer
    score = evaluator.evaluate(question_id, ideal)
    print(f"\nScore (correct answer '{ideal}'): {score}")

    # Test with wrong answer
    score = evaluator.evaluate(question_id, "wrong_answer")
    print(f"Score (wrong answer 'wrong_answer'): {score}")


if __name__ == "__main__":
    main()





