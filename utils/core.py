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

class QuizManager:
    """Utility class for managing interactive quizzes"""
    
    def __init__(self, session: str):
        self.session = session
        project_root = find_project_root()
        self.quizzes_path = project_root / 'utils/quizzes.json'
        
        # Load quizzes if the file exists, otherwise create a default empty structure
        if self.quizzes_path.exists():
            with open(self.quizzes_path, 'r') as f:
                self.quizzes = json.load(f)
        else:
            self.quizzes = {}
            print("📝 Note: quizzes.json not found. Using default empty quizzes.")
    
    def run_quiz(self, quiz_number: int):
        """Run a quiz by its number"""
        try:
            quiz_key = f"quiz_{quiz_number}"
            if self.session not in self.quizzes or quiz_key not in self.quizzes[self.session]:
                print(f"❌ Quiz {quiz_number} not found for session {self.session}")
                return
                
            quiz = self.quizzes[self.session][quiz_key]
            
            print("-" * 80)
            print(f"📋 {quiz['title']}")
            print("-" * 80)
            
            # Display the question and options
            print(quiz["question"])
            print()
            
            for i, option in enumerate(quiz["options"]):
                print(f"{chr(65 + i)}. {option}")
                
            print()
            
            # Get the answer from the user (wrapped in try-catch for interrupt handling)
            while True:
                try:
                    answer = input("Enter your answer (A, B, C, etc.): ").strip().upper()
                    if answer and answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(quiz["options"])]:
                        break
                    else:
                        print(f"Please enter a valid option (A-{chr(64 + len(quiz['options']))})")
                except KeyboardInterrupt:
                    print("\nQuiz aborted.")
                    return
                    
            # Check the answer
            correct_index = quiz["correct"]
            correct_letter = chr(65 + correct_index)
            
            if answer == correct_letter:
                print("\n✅ Correct!")
            else:
                print(f"\n❌ Incorrect. The correct answer is {correct_letter}.")
                
            # Display the explanation
            print("\n📚 Explanation:")
            print(quiz["explanation"])
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error running quiz: {str(e)}")

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
        print('-' * 80)
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
            
        # Handle weight initialization checks specific to SE03 exercise 2
        elif key.endswith("_weight_init") or key.endswith("_bias_init"):
            return self._check_weight_initialization(student_value, expected, key, exercise_data)
            
        # Handle metrics with min/max range or expected value with tolerance
        elif "expected" in expected or "min_val" in expected or "max_val" in expected:
            # Handle exact value with tolerance
            if "expected" in expected:
                tolerance = expected.get("tolerance", 1e-4)
                if abs(student_value - expected["expected"]) > tolerance:
                    self._print_error(
                        f"{key} has incorrect value. "
                        f"Expected {expected['expected']:.4f}, got {student_value:.4f}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            
            # Handle min/max range checks
            if "min_val" in expected and student_value < expected["min_val"]:
                self._print_error(
                    f"{key} is too small: {student_value:.4f}. Should be at least {expected['min_val']:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
            if "max_val" in expected and student_value > expected["max_val"]:
                self._print_error(
                    f"{key} is too large: {student_value:.4f}. Should be at most {expected['max_val']:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
            return True
        
        # Handle metrics specifically (can contain dictionaries or ranges)
        elif "metrics" in expected:
            return self._check_metrics(student_value, expected, key, exercise_data)

        return True

    def _check_weight_initialization(self, student_tensor, expected, key: str, exercise_data: dict) -> bool:
        """Validate weight or bias initialization"""
        if not isinstance(student_tensor, torch.Tensor):
            self._print_error(f"{key} should be a tensor")
            return False
        
        # Check mean close to zero for weights
        if key.endswith("_weight_init"):
            mean = student_tensor.mean().item()
            std = student_tensor.std().item()
            
            # Check if mean is close to zero (characteristic of proper initialization)
            if abs(mean) > 0.05:  # Allow small deviation from zero
                self._print_error(f"{key} mean should be close to zero, got {mean:.4f}")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            
            # Check if standard deviation is in reasonable range
            if "fc1" in key and (std < 0.1 or std > 0.5):
                self._print_error(
                    f"{key} standard deviation should be between 0.1 and 0.5 for proper He initialization, got {std:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            elif "fc2" in key and (std < 0.1 or std > 0.5):
                self._print_error(
                    f"{key} standard deviation should be between 0.1 and 0.5 for proper Xavier initialization, got {std:.4f}"
                )
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
        
        # Check if biases are initialized to zero
        elif key.endswith("_bias_init"):
            if not torch.allclose(student_tensor, torch.zeros_like(student_tensor), atol=1e-5):
                self._print_error(f"{key} should be initialized to zeros")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
        
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
            if not isinstance(student_model, torch.nn.Module):
                self._print_error(f"{key} has incorrect architecture")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False

        # Generic check for activation_type and optimizer type - compare class types generically
        if (key == "activation_type") or (key == "optimizer_type"):
            # For boolean expected value, we simply check if the type exists
            if expected.get("expected") is True:
                return True
            # Otherwise compare directly with expected type if available
            if "expected_type" in expected:
                expected_type = getattr(torch.nn, expected["expected_type"]) if key == "activation_type" else getattr(torch.optim, expected["expected_type"])  

                if not (student_model == expected_type or issubclass(student_model, expected_type)):
                    self._print_error(f"{key} has incorrect type. Expected {expected['expected_type']}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            return True
        

        # Generic check for weight initialization parameters
        if key in ["weight_init_kaiming", "weight_init_xavier", "bias_init_zeros"]:
            if not student_model:
                self._print_error(f"{key} initialization is missing")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            return True

        # Generic check for weight statistics
        if key.endswith("_weight_stats"):
            if not isinstance(student_model, dict):
                self._print_error(f"{key} should be a dictionary with mean and std")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
            
            # Check if mean is close to zero
            if "mean" not in student_model:
                self._print_error(f"{key} missing 'mean' field")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
            if "mean_near_zero" in expected and expected["mean_near_zero"]:
                if abs(student_model["mean"]) > 0.1:
                    self._print_error(f"{key} mean should be close to zero, got {student_model['mean']:.4f}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                
            # Check if std is in reasonable range
            if "std" not in student_model:
                self._print_error(f"{key} missing 'std' field")
                self._show_relevant_hint(exercise_data["hints"], key)
                return False
                
            if "std_range" in expected and len(expected["std_range"]) == 2:
                min_std, max_std = expected["std_range"]
                if student_model["std"] < min_std or student_model["std"] > max_std:
                    self._print_error(f"{key} std should be between {min_std} and {max_std}, got {student_model['std']:.4f}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                
            return True

        # Generic model validation for any session/exercise
        if key == "model":
            # Check if model has expected input/output dimensions
            if hasattr(expected, "input_size") and hasattr(student_model, "fc1") and hasattr(student_model.fc1, "in_features"):
                if student_model.fc1.in_features != expected["input_size"]:
                    self._print_error(f"Model input size should be {expected['input_size']}, got {student_model.fc1.in_features}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                
            # Check for appropriate hidden layer size to avoid overfitting
            if "max_hidden_size" in expected and hasattr(student_model, "fc1") and hasattr(student_model.fc1, "out_features"):
                if student_model.fc1.out_features > expected["max_hidden_size"]:
                    self._print_error(f"Hidden layer size {student_model.fc1.out_features} is too large (max: {expected['max_hidden_size']}) and may overfit")
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
            for metric_name, expected_range in metrics.items():
                if metric_name not in student_value:
                    self._print_error(f"{key} missing metric {metric_name}")
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
                
                # Handle range as (min, max) tuple
                if isinstance(expected_range, (list, tuple)) and len(expected_range) == 2:
                    min_val, max_val = expected_range
                    if not (min_val <= student_value[metric_name] <= max_val):
                        self._print_error(
                            f"{key} {metric_name} should be between {min_val} and {max_val}, got {student_value[metric_name]:.4f}"
                        )
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False
                # Handle exact value with tolerance
                elif isinstance(expected_range, dict) and "expected" in expected_range:
                    expected_val = expected_range["expected"]
                    tolerance = expected_range.get("tolerance", 1e-4)
                    if abs(student_value[metric_name] - expected_val) > tolerance:
                        self._print_error(
                            f"{key} {metric_name} should be close to {expected_val}, got {student_value[metric_name]:.4f}"
                        )
                        self._show_relevant_hint(exercise_data["hints"], key)
                        return False
        
        # Handle single metric value with range
        else:
            # Check if metrics contains a "value" entry with (min, max) format
            if "value" in metrics and isinstance(metrics["value"], (list, tuple)) and len(metrics["value"]) == 2:
                min_val, max_val = metrics["value"]
                if not (min_val <= student_value <= max_val):
                    self._print_error(
                        f"{key} should be between {min_val} and {max_val}, got {student_value:.4f}"
                    )
                    self._show_relevant_hint(exercise_data["hints"], key)
                    return False
            # Handle exact value with tolerance
            elif "expected" in metrics:
                expected_val = metrics["expected"]
                tolerance = metrics.get("tolerance", 1e-4)
                if abs(student_value - expected_val) > tolerance:
                    self._print_error(
                        f"{key} should be close to {expected_val}, got {student_value:.4f}"
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