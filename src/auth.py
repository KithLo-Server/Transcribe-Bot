"""Security-question authentication with a persisted lockout counter."""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

from storage import atomic_write_json, read_json

logger = logging.getLogger(__name__)


def load_questions(path: str) -> List[Dict[str, List[str]]]:
    """Loads and validates the security question file.

    Expected format:
        [{"question": "...", "answers": ["one accepted answer", "..."]}, ...]

    The singular "answer" key is also accepted for compatibility.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("Security question file must contain a non-empty JSON array.")

    questions = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be an object.")
        question = item.get("question")
        answers = item.get("answers", item.get("answer"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {index} has no valid question text.")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"Question {index} must have a non-empty answer array.")
        if not all(isinstance(answer, str) and answer.strip() for answer in answers):
            raise ValueError(f"Question {index} contains an invalid answer.")
        questions.append(
            {
                "question": question.strip(),
                "answer": [answer for answer in answers],
            }
        )
    return questions


def normalize_answer(value: str) -> str:
    return " ".join(value.split()).casefold()


@dataclass
class AuthResult:
    status: str
    question: Optional[str] = None
    question_number: int = 0
    question_total: int = 0
    accepted: bool = False
    attempts_remaining: int = 0
    lock_remaining: int = 0


class AuthManager:
    def __init__(
        self,
        questions: Sequence[Dict[str, List[str]]],
        state_file: str,
        max_failures: int = 5,
        lockout_seconds: int = 3600,
        now_fn: Callable[[], float] = time.time,
    ):
        self.questions = list(questions)
        self.normalized_answers = [
            {normalize_answer(answer) for answer in item["answer"]}
            for item in self.questions
        ]
        self.state_file = Path(state_file)
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self.now_fn = now_fn
        self._state, self._authenticated = self._load_state()

    # ----------------- Persistence -----------------
    def _load_state(self):
        data = read_json(self.state_file)
        if not isinstance(data, dict):
            return {}, set()

        if data.get("version") == 2 and isinstance(data.get("users"), dict):
            users = data["users"]
            authenticated = data.get("authenticated", [])
        else:
            # Legacy format was a flat mapping of user id to lockout data.
            users = data
            authenticated = []

        clean_users = {
            str(user_id): entry
            for user_id, entry in users.items()
            if isinstance(entry, dict)
        }
        authenticated_ids: Set[int] = set()
        if isinstance(authenticated, list):
            for value in authenticated:
                try:
                    authenticated_ids.add(int(value))
                except (TypeError, ValueError):
                    logger.warning("Ignoring invalid authenticated user: %r", value)
        return clean_users, authenticated_ids

    def _save_state(self) -> None:
        atomic_write_json(
            self.state_file,
            {
                "version": 2,
                "users": self._state,
                "authenticated": sorted(self._authenticated),
            },
        )

    def _entry(self, user_id: int) -> Dict:
        key = str(user_id)
        entry = self._state.setdefault(
            key, {"failures": 0, "locked_until": 0.0, "question_index": 0}
        )
        entry.setdefault("failures", 0)
        entry.setdefault("locked_until", 0.0)
        entry.setdefault("question_index", 0)
        return entry

    def _clear_expired_lock(self, entry: Dict) -> None:
        if entry["locked_until"] and self.now_fn() >= entry["locked_until"]:
            entry["locked_until"] = 0.0
            entry["failures"] = 0
            entry["question_index"] = 0

    def _lock_remaining(self, entry: Dict) -> int:
        remaining = int(entry["locked_until"] - self.now_fn())
        return max(0, remaining)

    # ----------------- Public API -----------------
    def is_authenticated(self, user_id: int) -> bool:
        return user_id in self._authenticated

    def is_locked(self, user_id: int) -> int:
        entry = self._entry(user_id)
        self._clear_expired_lock(entry)
        return self._lock_remaining(entry) if entry["locked_until"] else 0

    def start(self, user_id: int) -> AuthResult:
        entry = self._entry(user_id)
        self._clear_expired_lock(entry)
        remaining = self._lock_remaining(entry)
        if remaining:
            return AuthResult("locked", lock_remaining=remaining)

        entry["question_index"] = 0
        self._save_state()
        return self._question_result(entry, accepted=False)

    def submit_answer(self, user_id: int, answer: str) -> AuthResult:
        entry = self._entry(user_id)
        self._clear_expired_lock(entry)
        remaining = self._lock_remaining(entry)
        if remaining:
            return AuthResult("locked", lock_remaining=remaining)

        index = int(entry.get("question_index", 0))
        if index >= len(self.questions):
            index = 0
            entry["question_index"] = 0

        if normalize_answer(answer) in self.normalized_answers[index]:
            index += 1
            entry["failures"] = 0
            if index >= len(self.questions):
                entry["question_index"] = 0
                self._authenticated.add(user_id)
                self._save_state()
                return AuthResult(
                    "authenticated",
                    accepted=True,
                    question_total=len(self.questions),
                )
            entry["question_index"] = index
            self._save_state()
            return self._question_result(entry, accepted=True)

        entry["failures"] = int(entry.get("failures", 0)) + 1
        if entry["failures"] >= self.max_failures:
            entry["locked_until"] = self.now_fn() + self.lockout_seconds
            entry["failures"] = 0
            entry["question_index"] = 0
            self._save_state()
            return AuthResult(
                "locked",
                lock_remaining=self.lockout_seconds,
                attempts_remaining=0,
            )

        self._save_state()
        return AuthResult(
            "question",
            question=self.questions[index]["question"],
            question_number=index + 1,
            question_total=len(self.questions),
            accepted=False,
            attempts_remaining=self.max_failures - entry["failures"],
        )

    def _question_result(self, entry: Dict, accepted: bool) -> AuthResult:
        index = int(entry["question_index"])
        return AuthResult(
            "question",
            question=self.questions[index]["question"],
            question_number=index + 1,
            question_total=len(self.questions),
            accepted=accepted,
            attempts_remaining=self.max_failures - int(entry.get("failures", 0)),
        )
