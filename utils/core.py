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

        """Initialize checker with answers from JSON"""
        project_root = find_project_root()
        self.solutions_path = project_root / 'utils/solutions.json'
        with open(self.solutions_path, 'r') as f:
            self.answers = json.load(f)
    
    @staticmethod
    def _print_success(msg: str) -> None:
        print(f"✅ {msg}")
    
    @staticmethod
    def _print_error(msg: str) -> None:
        print(f"❌ {msg}")
    
    @staticmethod
    def _print_hint(msg: str) -> None:
        print(f"💡 Hint: {msg}")
    
    def check_exercise(self, exercise: str, student_answer: dict) -> None:
        """Check student answer against solution"""
        exercise_data = self.answers[self.session][f"exercise_{exercise}"]
        all_correct = True
        
        # Check each part of the answer
        for key, expected in exercise_data["answers"].items():
            if key not in student_answer:
                self._print_error(f"Missing {key}")
                continue
                
            student_value = student_answer[key]
            
            # Check value
            try:
                if expected["dtype"].startswith("torch."):
                    # Tensor comparison
                    expected_tensor = torch.tensor(expected["value"], 
                                                dtype=getattr(torch, expected["dtype"].split('.')[1]))
                    if torch.allclose(student_value, expected_tensor):
                        self._print_success(f"{key} is correct!")
                    else:
                        self._print_error(f"{key} is incorrect")
                        self._print_hint(exercise_data["hints"][0])
                        all_correct = False
                else:
                    # Regular value comparison
                    if student_value == expected["value"]:
                        self._print_success(f"{key} is correct!")
                    else:
                        self._print_error(f"{key} is incorrect")
                        self._print_hint(exercise_data["hints"][0])
                        all_correct = False
            except Exception as e:
                self._print_error(f"Error checking {key}: {str(e)}")
                all_correct = False
        
        if all_correct:
            print("\n🎉 Excellent! All parts are correct!")
    
    def display_hints(self, exercise: str) -> None:
        """Display hints for the exercise"""
        exercise_data = self.answers[self.session][f"exercise_{exercise}"]
        print("\n💡 Hints:")
        for hint in exercise_data["hints"]:
            print(f"- {hint}")