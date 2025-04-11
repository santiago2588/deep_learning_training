import torch
import json
from pathlib import Path

def find_project_root() -> Path:
    current_path = Path(__file__).resolve()
    while current_path != current_path.root:
        if (current_path / 'utils').exists():  # Check if 'utils' directory exists
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root not found")

class ExerciseChecker:
    """Utility class for checking workshop exercises"""
    
    def __init__(self, session: str):
        self.session = session
        project_root = find_project_root()
        self.solutions_path = project_root / 'utils/solutions.json'
        with open(self.solutions_path, 'r') as f:
            self.answers = json.load(f)
        self.hints_shown = set()

    @staticmethod
    def _print_success(msg: str) -> None:
        print(f"✅ {msg}")
    
    @staticmethod
    def _print_error(msg: str) -> None:
        print(f"❌ {msg}")
    
    @staticmethod
    def _print_hint(msg: str) -> None:
        print(f"💡 Hint: {msg}")
    
    def check_exercise(self, exercise: int, student_answer: dict) -> None:
        """
        Check student answer against solution.
        Handles various types including tensors, models and metrics.
        """
        exercise_key = f"exercise_{exercise}"
        exercise_data = self.answers[self.session][exercise_key]
        all_correct = True
        
        for key, expected in exercise_data["answers"].items():
            if key not in student_answer:
                self._print_error(f"Missing {key}")
                self._show_relevant_hint(exercise_data["hints"], key)
                all_correct = False
                continue

            try:
                value_correct = self._check_value(
                    student_value=student_answer[key],
                    expected=expected,
                    key=key,
                    exercise_data=exercise_data
                )
                
                if value_correct:
                    self._print_success(f"{key} is correct!")
                else:
                    all_correct = False

            except Exception as e:
                self._print_error(f"Error checking {key}: {str(e)}")
                self._show_relevant_hint(exercise_data["hints"], key)
                all_correct = False

        if all_correct:
            print("\n🎉 Excellent! All parts are correct!")

    def _check_value(self, student_value, expected, key: str, exercise_data: dict) -> bool:
        """Check a single value against expected solution"""
        
        # Handle functions (like sigmoid)
        if "test_cases" in expected:
            return self._check_function(student_value, expected, key, exercise_data)
            
        # Handle regular tensors
        elif expected.get("dtype", "").startswith("torch."):
            return self._check_tensor(student_value, expected, key, exercise_data)
            
        # Handle PyTorch models
        elif isinstance(student_value, torch.nn.Module):
            return self._check_model(student_value, expected, key, exercise_data)
            
        # Handle tensor shapes (special case for torch.Size objects)
        elif isinstance(student_value, torch.Size):
            if "shape" in expected:
                expected_shape = tuple(expected["shape"])
                if student_value != expected_shape:
                    self._print_error(
                        f"{key} has wrong shape. Expected {expected_shape}, got {student_value}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            return True
            
        # Handle metrics with tolerance
        elif "expected" in expected:
            tolerance = expected.get("tolerance", 1e-4)
            if abs(student_value - expected["expected"]) > tolerance:
                self._print_error(
                    f"{key} has incorrect value. "
                    f"Expected {expected['expected']:.4f}, got {student_value:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            return True

        return True

    def _check_function(self, student_func, expected, key: str, exercise_data: dict) -> bool:
        """Validate function implementation using test cases"""
        if not callable(student_func):
            self._print_error(f"{key} should be a function")
            return False

        test_cases = expected.get("test_cases", [])
        tolerance = expected.get("tolerance", 1e-4)

        for case in test_cases:
            input_data = case["input"]
            expected_val = case["expected"]
            
            try:
                # Handle MSE loss function test case (special format for SE02.exercise_3)
                if isinstance(input_data[0], dict) and "predictions" in input_data[0]:
                    predictions = torch.tensor(input_data[0]["predictions"], dtype=torch.float32)
                    targets = torch.tensor(input_data[0]["targets"], dtype=torch.float32)
                    result = student_func(predictions, targets).item()
                else:
                    # Regular function test case
                    input_val = torch.tensor(input_data, dtype=torch.float32)
                    result = student_func(input_val).item()
                    
                if abs(result - expected_val) > tolerance:
                    self._print_error(
                        f"{key} failed test case: expected {expected_val:.4f}, got {result:.4f}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            except Exception as e:
                self._print_error(f"Error testing {key}: {str(e)}")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
        
        return True

    def _check_tensor(self, student_tensor, expected, key: str, exercise_data: dict) -> bool:
        """Validate tensor properties and values"""
        if not isinstance(student_tensor, torch.Tensor):
            self._print_error(f"{key} should be a tensor")
            return False

        # Check shape if specified
        if "shape" in expected:
            if student_tensor.shape != tuple(expected["shape"]):
                self._print_error(
                    f"{key} has wrong shape. Expected {tuple(expected['shape'])}, got {student_tensor.shape}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check value range if specified
        if "min_val" in expected and "max_val" in expected:
            if not (expected["min_val"] <= student_tensor.min().item() <= expected["max_val"] and 
                   expected["min_val"] <= student_tensor.max().item() <= expected["max_val"]):
                self._print_error(
                    f"{key} values should be between {expected['min_val']} and {expected['max_val']}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check exact values if specified
        if "value" in expected:
            expected_tensor = torch.tensor(
                expected["value"], 
                dtype=getattr(torch, expected["dtype"].split('.')[1])
            )
            if not torch.allclose(student_tensor, expected_tensor, rtol=1e-3):
                self._print_error(f"{key} has incorrect values")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        return True

    def _check_model(self, student_model, expected, key: str, exercise_data: dict) -> bool:
        """Validate neural network model properties"""
        # Check architecture
        if "architecture" in expected:
            if not isinstance(student_model, getattr(torch.nn, expected["architecture"])):
                self._print_error(f"{key} has incorrect architecture")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Check layer properties
        if "layers" in expected:
            for layer_name, layer_props in expected["layers"].items():
                if not hasattr(student_model, layer_name):
                    self._print_error(f"{key} missing layer {layer_name}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                
                layer = getattr(student_model, layer_name)
                for prop_name, prop_value in layer_props.items():
                    if not hasattr(layer, prop_name) or getattr(layer, prop_name) != prop_value:
                        self._print_error(f"{key} layer {layer_name} has incorrect {prop_name}")
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False

        return True

    def _check_metrics(self, student_value, expected, key: str, exercise_data: dict) -> bool:
        """Validate training metrics like loss and accuracy"""
        metrics = expected["metrics"]
        
        # Handle dictionary of metrics
        if isinstance(student_value, dict):
            for metric_name, (min_val, max_val) in metrics.items():
                if metric_name not in student_value:
                    self._print_error(f"{key} missing metric {metric_name}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                
                if not (min_val <= student_value[metric_name] <= max_val):
                    self._print_error(
                        f"{key} {metric_name} should be between {min_val} and {max_val}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
        
        # Handle single metric value
        else:
            min_val, max_val = metrics["value"]
            if not (min_val <= student_value <= max_val):
                self._print_error(
                    f"{key} should be between {min_val} and {max_val}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        return True

    def _show_relevant_hint(self, hints: list, key: str) -> None:
        """Show context-aware hints for the current error"""
        # Try to find a specific hint for this key
        specific_hint = next(
            (hint for hint in hints if key.lower() in hint.lower()), 
            None
        )
        
        if specific_hint and specific_hint not in self.hints_shown:
            self._print_hint(specific_hint)
            self.hints_shown.add(specific_hint)
        elif hints and hints[0] not in self.hints_shown:
            self._print_hint(hints[0])
            self.hints_shown.add(hints[0])

    def display_hints(self, exercise: str) -> None:
        """Display hints for the exercise"""
        exercise_data = self.answers[self.session][f"exercise_{exercise}"]
        print("\n💡 Hints:")
        for hint in exercise_data["hints"]:
            print(f"- {hint}")