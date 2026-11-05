"""Unit tests for security-question authentication and lockouts."""

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from auth import AuthManager, load_questions


QUESTIONS = [
    {"question": "First question?", "answer": ["Alpha", "A"]},
    {"question": "Second question?", "answer": ["Beta"]},
]


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.tempdir.name) / "auth.json"
        self.now = 1000.0
        self.manager = AuthManager(
            QUESTIONS,
            str(self.state_file),
            max_failures=5,
            lockout_seconds=3600,
            now_fn=lambda: self.now,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_answers_are_case_insensitive_and_trimmed(self):
        result = self.manager.start(1)
        self.assertEqual(result.status, "question")
        self.assertEqual(result.question, "First question?")

        result = self.manager.submit_answer(1, "  a  ")
        self.assertTrue(result.accepted)
        self.assertEqual(result.question, "Second question?")
        self.assertFalse(self.manager.is_authenticated(1))

        result = self.manager.submit_answer(1, "BETA")
        self.assertEqual(result.status, "authenticated")
        self.assertTrue(self.manager.is_authenticated(1))

    def test_all_questions_are_required(self):
        self.manager.start(1)
        self.manager.submit_answer(1, "Alpha")
        self.assertFalse(self.manager.is_authenticated(1))
        self.manager.submit_answer(1, "Beta")
        self.assertTrue(self.manager.is_authenticated(1))

    def test_wrong_answer_reasks_same_question(self):
        self.manager.start(1)
        result = self.manager.submit_answer(1, "wrong")
        self.assertFalse(result.accepted)
        self.assertEqual(result.question, "First question?")
        self.assertEqual(result.attempts_remaining, 4)

    def test_five_wrong_answers_lock_for_an_hour(self):
        self.manager.start(1)
        for remaining in (4, 3, 2, 1):
            result = self.manager.submit_answer(1, "wrong")
            self.assertEqual(result.status, "question")
            self.assertEqual(result.attempts_remaining, remaining)

        result = self.manager.submit_answer(1, "wrong")
        self.assertEqual(result.status, "locked")
        self.assertEqual(self.manager.is_locked(1), 3600)

        self.now += 3601
        self.assertEqual(self.manager.is_locked(1), 0)
        result = self.manager.start(1)
        self.assertEqual(result.status, "question")
        self.assertEqual(result.question, "First question?")

    def test_success_resets_failure_counter(self):
        self.manager.start(1)
        for _ in range(4):
            self.manager.submit_answer(1, "wrong")

        result = self.manager.submit_answer(1, "Alpha")
        self.assertTrue(result.accepted)
        self.assertFalse(self.manager.is_authenticated(1))
        result = self.manager.submit_answer(1, "Beta")
        self.assertEqual(result.status, "authenticated")

    def test_lockout_is_persisted(self):
        self.manager.start(7)
        for _ in range(5):
            self.manager.submit_answer(7, "wrong")

        reloaded = AuthManager(
            QUESTIONS,
            str(self.state_file),
            max_failures=5,
            lockout_seconds=3600,
            now_fn=lambda: self.now,
        )
        self.assertEqual(reloaded.is_locked(7), 3600)
        self.assertEqual(reloaded.start(7).status, "locked")

    def test_authenticated_status_is_persisted(self):
        self.manager.start(3)
        self.manager.submit_answer(3, "Alpha")
        result = self.manager.submit_answer(3, "Beta")
        self.assertEqual(result.status, "authenticated")

        reloaded = AuthManager(
            QUESTIONS,
            str(self.state_file),
            max_failures=5,
            lockout_seconds=3600,
            now_fn=lambda: self.now,
        )
        self.assertTrue(reloaded.is_authenticated(3))
        self.assertFalse(reloaded.is_authenticated(4))

    def test_legacy_auth_file_is_migrated(self):
        legacy = {
            "9": {"failures": 2, "locked_until": 0.0, "question_index": 1}
        }
        self.state_file.write_text(json.dumps(legacy), encoding="utf-8")

        manager = AuthManager(
            QUESTIONS,
            str(self.state_file),
            max_failures=5,
            lockout_seconds=3600,
            now_fn=lambda: self.now,
        )
        self.assertEqual(manager.start(9).question, "First question?")

        saved = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["version"], 2)
        self.assertIn("9", saved["users"])
        self.assertEqual(saved["authenticated"], [])


class QuestionFileTests(unittest.TestCase):
    def _write(self, path: Path, data) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_valid_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "questions.json"
            self._write(path, QUESTIONS)
            questions = load_questions(str(path))
            self.assertEqual(len(questions), 2)
            self.assertEqual(questions[0]["answer"], ["Alpha", "A"])

    def test_answers_key_is_supported(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "questions.json"
            self._write(
                path,
                [{"question": "Q", "answers": ["one", "two"]}],
            )
            questions = load_questions(str(path))
            self.assertEqual(questions[0]["answer"], ["one", "two"])

    def test_invalid_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "questions.json"
            for invalid in (
                [],
                [{"question": "Q", "answer": []}],
                [{"question": "", "answer": ["A"]}],
                [{"question": "Q", "answer": "A"}],
                [{"question": "Q", "answer": [""]}],
            ):
                self._write(path, invalid)
                with self.assertRaises(ValueError):
                    load_questions(str(path))


if __name__ == "__main__":
    unittest.main()
