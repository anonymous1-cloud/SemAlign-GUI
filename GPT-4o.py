import openai
from openai import OpenAI
import base64
from PIL import Image
import io
import json
from typing import List, Dict, Tuple
import numpy as np


class GPT4oEvaluator:
    """
    GPT-4o evaluator for zero-shot GUI change understanding
    Based on: Islam, M.J., et al. (2025). GPT-4o: Advancing Multimodal Understanding.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_pil_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_gui_changes(self,
                            ref_image_path: str,
                            tar_image_path: str,
                            prompt_template: str = None) -> Dict:
        """
        Analyze GUI changes using GPT-4o

        Args:
            ref_image_path: Path to reference GUI screenshot
            tar_image_path: Path to target GUI screenshot
            prompt_template: Custom prompt template

        Returns:
            Dictionary containing analysis results
        """

        # Encode images
        ref_image_base64 = self.encode_image_to_base64(ref_image_path)
        tar_image_base64 = self.encode_image_to_base64(tar_image_path)

        # Default prompt for GUI change analysis
        if prompt_template is None:
            prompt_template = """You are an expert in GUI/UI analysis and software testing. 
            Analyze the two GUI screenshots (Reference and Target) and identify all changes.

            Please provide:
            1. List of visual changes (additions, removals, modifications)
            2. Semantic description of changes
            3. Likelihood of this being an intentional update vs. a bug
            4. Suggested test cases for this change

            Format your response as JSON with the following structure:
            {
                "changes": [
                    {
                        "type": "addition|removal|modification|movement",
                        "component": "button|text|image|input_field|etc",
                        "description": "detailed description",
                        "location": "approximate location description",
                        "confidence": 0.0-1.0
                    }
                ],
                "semantic_summary": "overall summary of changes",
                "intentional_update_score": 0.0-1.0,
                "test_cases": ["test case 1", "test case 2", ...]
            }"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_template},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{ref_image_base64}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{tar_image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.2
            )

            # Parse response
            result_text = response.choices[0].message.content

            # Try to extract JSON from response
            try:
                # Find JSON in response
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                json_str = result_text[start_idx:end_idx]
                result_json = json.loads(json_str)
                return result_json
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                return {"raw_response": result_text}

        except Exception as e:
            return {"error": str(e)}

    def batch_evaluate(self,
                       test_cases: List[Tuple[str, str, str]],
                       output_file: str = "gpt4o_results.json") -> List[Dict]:
        """
        Batch evaluate multiple test cases

        Args:
            test_cases: List of (ref_image_path, tar_image_path, ground_truth_description)
            output_file: Path to save results

        Returns:
            List of evaluation results
        """
        results = []

        for i, (ref_path, tar_path, gt_desc) in enumerate(test_cases):
            print(f"Processing case {i + 1}/{len(test_cases)}")

            result = self.analyze_gui_changes(ref_path, tar_path)

            # Add ground truth for evaluation
            result["ground_truth"] = gt_desc
            result["ref_image"] = ref_path
            result["tar_image"] = tar_path

            results.append(result)

            # Save intermediate results
            if (i + 1) % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

        # Save final results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def evaluate_change_detection(self,
                                  predictions: List[Dict],
                                  ground_truths: List[Dict],
                                  metrics: List[str] = None) -> Dict:
        """
        Evaluate GPT-4o's change detection performance

        Args:
            predictions: List of prediction dictionaries from GPT-4o
            ground_truths: List of ground truth dictionaries
            metrics: List of metrics to compute

        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'f1', 'accuracy']

        scores = {}

        # Implement evaluation metrics
        # Note: This requires alignment between GPT-4o's output and ground truth
        # You may need to implement custom logic based on your evaluation framework

        for metric in metrics:
            if metric == 'precision':
                # Calculate precision
                scores['precision'] = self._calculate_precision(predictions, ground_truths)
            elif metric == 'recall':
                # Calculate recall
                scores['recall'] = self._calculate_recall(predictions, ground_truths)
            elif metric == 'f1':
                # Calculate F1 score
                precision = scores.get('precision', self._calculate_precision(predictions, ground_truths))
                recall = scores.get('recall', self._calculate_recall(predictions, ground_truths))
                if precision + recall > 0:
                    scores['f1'] = 2 * precision * recall / (precision + recall)
                else:
                    scores['f1'] = 0
            elif metric == 'accuracy':
                # Calculate accuracy
                scores['accuracy'] = self._calculate_accuracy(predictions, ground_truths)

        return scores

    def _calculate_precision(self, predictions, ground_truths):
        # Simplified precision calculation
        # Implement based on your specific evaluation criteria
        correct_predictions = 0
        total_predictions = 0

        for pred, gt in zip(predictions, ground_truths):
            # Compare predicted changes with ground truth
            # This is a simplified version - adapt to your needs
            pred_changes = pred.get('changes', [])
            gt_changes = gt.get('changes', [])

            # Count matches
            for p_change in pred_changes:
                if self._is_change_match(p_change, gt_changes):
                    correct_predictions += 1
                total_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def _is_change_match(self, pred_change, gt_changes, threshold=0.7):
        """Check if predicted change matches any ground truth change"""
        # Implement matching logic based on your criteria
        for gt_change in gt_changes:
            # Compare type, component, location, etc.
            if (pred_change.get('type') == gt_change.get('type') and
                    pred_change.get('component') == gt_change.get('component')):
                return True
        return False


# Usage example
def run_gpt4o_evaluation(api_key, test_data_path):
    evaluator = GPT4oEvaluator(api_key=api_key)

    # Load test data
    with open(test_data_path, 'r') as f:
        test_cases = json.load(f)

    # Run evaluation
    results = evaluator.batch_evaluate(test_cases)

    # Calculate metrics
    metrics = evaluator.evaluate_change_detection(
        predictions=results,
        ground_truths=test_cases  # Assuming test_cases contain ground truth
    )

    print("GPT-4o Evaluation Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    return results, metrics

# Note: GPT-4o API requires valid API key and may incur costs