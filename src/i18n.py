"""Localization for user-facing Telegram messages.

Locales are resolved from Telegram's ``language_code``:

* ``zh`` and Simplified Chinese variants use ``zh-Hans``.
* Traditional Chinese and Hong Kong variants use ``zh-Hant``.
* Everything else falls back to English.
"""

from typing import Any, Dict, Optional

DEFAULT_LOCALE = "en"
LOCALES = ("en", "zh-Hans", "zh-Hant")

_LANGUAGE_KEYS = {
    "Cantonese": "language.cantonese",
    "Mandarin": "language.mandarin",
    "English": "language.english",
}

_FORMAT_KEYS = {
    "Text": "format.text",
    "SRT": "format.srt",
}

_ZH_KEYS = {
    "Traditional Chinese": "zh.traditional",
    "Simplified Chinese": "zh.simplified",
}

_EN = {
    "auth.authenticated": (
        "You are authenticated. Send an audio/video file or a .txt/.srt "
        "document to begin."
    ),
    "auth.locked": (
        "Too many failed attempts. Try again in about {minutes} minute(s)."
    ),
    "auth.question": (
        "{prefix}Security question {number}/{total}:\n\n"
        "{question}\n\nReply with your answer."
    ),
    "auth.correct_prefix": "Correct.\n\n",
    "auth.incorrect_prefix": (
        "Incorrect. {attempts} attempt(s) remaining.\n\n"
    ),
    "auth.correct_authenticated": (
        "Correct. You are authenticated. Send an audio/video file or a "
        ".txt/.srt document to begin."
    ),
    "auth.status_required": (
        "Please authenticate with /start before using /status."
    ),
    "auth.upload_required": (
        "Please authenticate with /start before uploading a file."
    ),
    "auth.callback_required": (
        "Please authenticate with /start before using this."
    ),
    "status.none_admin": "No active jobs.",
    "status.none_user": "You have no active jobs.",
    "status.header_admin": "All active jobs ({count}):",
    "status.header_user": "Your active jobs ({count}):",
    "status.line": (
        "{index}. {user_prefix}{file_name} - {status} "
        "({lang}, {fmt}, job {job_id})"
    ),
    "status.user_prefix": "[user {user_id}] ",
    "status.pending": "Preparing",
    "status.queued": "Queued",
    "status.processing": "Transcribing",
    "status.raw_cache": "Waiting for transcription cache choice",
    "status.refine_cache": "Waiting for refinement cache choice",
    "status.refining": "Refining",
    "status.refine_failed": "Refinement failed (retry available)",
    "text.prompt_or_skip": "Send a prompt or press Skip.",
    "wizard.use_refinement": "Use AI refinement?",
    "wizard.char_limit_invalid": (
        "Enter a number between 10 and 100 for the target characters "
        "per subtitle line."
    ),
    "wizard.use_buttons": "Please use the buttons above.",
    "wizard.send_file": (
        "Send an audio/video file or a .txt/.srt document to begin."
    ),
    "upload.could_not_read": "Could not read that attachment.",
    "upload.too_large": (
        "That file is larger than the configured upload limit."
    ),
    "upload.download_failed": "Could not download that file: {error}",
    "upload.mixed": (
        "Do not mix media files with .txt/.srt uploads in one batch. "
        "Finish or cancel the current batch first."
    ),
    "upload.added": (
        "Added {file_name} to the current batch "
        "({count} files). The same settings will apply "
        "to every file in this batch."
    ),
    "upload.received": (
        "Received {file_name}.\n\n"
        "The settings you choose will apply to every file uploaded before "
        "this wizard finishes.\n\nChoose the language."
    ),
    "cache.unknown": "Unknown cache choice.",
    "cache.inactive": "This cache choice is no longer active.",
    "cache.resolve_failed": "Could not resolve cache choice: {error}",
    "cache.retry_failed": "Could not retry the refinement: {error}",
    "cache.no_files": (
        "No files are waiting for settings. Send a file again."
    ),
    "confirm.inactive": "This confirmation is no longer active.",
    "confirm.edit_language": (
        "Choose the language. Uploaded files in this batch are kept."
    ),
    "confirm.cancelled": "Cancelled. The uploaded files were removed.",
    "confirm.unknown": "Unknown confirmation choice.",
    "language.unknown": "Unknown language selection.",
    "c2w.prompt": "Convert spoken Cantonese to written Chinese?",
    "char_limit.prompt": (
        "Enter the target character count per subtitle line "
        "(10-100, for example 30)."
    ),
    "format.language_chosen": (
        "Language: {lang}\n\nChoose the output format."
    ),
    "format.choose_language_first": "Choose a language first.",
    "format.unknown": "Unknown output format.",
    "prompt.whisper": (
        "Send the initial prompt for faster-whisper, or press Skip."
    ),
    "zh.prompt": "Choose the Chinese output format.",
    "zh.unknown": "Unknown Chinese format.",
    "selection.unknown": "Unknown selection.",
    "manual.upload_prompt": (
        "Manual AI refinement. Send a .txt or .srt file to refine. You will "
        "copy each prompt into your preferred AI and send its response back."
    ),
    "manual.text_only": (
        "Manual refinement accepts only .txt or .srt files."
    ),
    "manual.single_file": (
        "Manual refinement handles one file at a time. Finish or cancel the "
        "current refinement first."
    ),
    "manual.invalid_response": (
        "Send the AI response as a message or a .txt/.srt file."
    ),
    "manual.invalid_srt": (
        "That response is not valid SRT. Send the AI response again."
    ),
    "manual.empty_response": "The AI response is empty. Send it again.",
    "manual.read_failed": "Could not read the response file: {error}",
    "manual.start_failed": "Could not start manual refinement: {error}",
    "manual.prompt_header": (
        "Manual refinement prompt {step}/{total}. Copy the prompt into your "
        "AI, then send its full response here as a message or a .txt/.srt "
        "file."
    ),
    "manual.plan_header": (
        "Manual structure-planning prompt. Copy the prompt into your AI, then "
        "send its full JSON response here as a message or a .txt file."
    ),
    "manual.plan_applied": (
        "Structure plan accepted. Preparing {count} refinement prompt(s)."
    ),
    "manual.plan_fallback": (
        "The structure plan could not be read; using automatic chunking. "
        "Preparing {count} refinement prompt(s)."
    ),
    "manual.exceed_unavailable": (
        "There is no manual refinement prompt waiting for a response."
    ),
    "manual.exceed_impossible": (
        "This prompt cannot be split further. Try another AI or send the "
        "response you have."
    ),
    "manual.exceed_split": (
        "The current prompt was split into {count} smaller prompts. Send the "
        "response to the prompt below."
    ),
    "manual.merge_failed": (
        "Could not assemble the final refinement: {error}"
    ),
    "manual.send_failed": (
        "Could not send manual refinement output: {error}"
    ),
    "manual.complete": (
        "Manual refinement complete. The final refined text is attached."
    ),
    "manual.restart_resume": (
        "Bot restarted; resuming manual refinement."
    ),
    "summary.review": "Review your settings before submitting.",
    "summary.files": "Files: {count}",
    "summary.language": "Language: {lang}",
    "summary.output_from_extension": (
        "Output format: {formats} (from file extension)"
    ),
    "summary.output": "Output format: {fmt}",
    "summary.zh_format": "Chinese format: {zh}",
    "summary.initial_prompt": "Initial prompt: {prompt}",
    "summary.none": "(none)",
    "summary.refine_required": (
        "AI refinement: Yes (required for text uploads)"
    ),
    "summary.manual_refine": (
        "AI refinement: Manual (you supply the AI responses)"
    ),
    "summary.refine": "AI refinement: {value}",
    "summary.c2w": "Cantonese to written Chinese: {value}",
    "summary.char_limit": "Characters per subtitle line: {count}",
    "summary.footer": "Confirm, edit the settings, or cancel.",
    "summary.yes": "Yes",
    "summary.no": "No",
    "finalize.missing": (
        "Some settings are missing. Send the file(s) again."
    ),
    "finalize.submitted": "Submitted {count} files.",
    "language.cantonese": "Cantonese",
    "language.mandarin": "Mandarin",
    "language.english": "English",
    "format.text": "Text",
    "format.srt": "SRT",
    "zh.traditional": "Traditional Chinese",
    "zh.simplified": "Simplified Chinese",
    "button.skip": "Skip",
    "button.yes": "Yes",
    "button.no": "No",
    "button.confirm": "Confirm",
    "button.edit": "Edit",
    "button.cancel": "Cancel",
    "button.use_transcription": "Use existing transcription",
    "button.start_anew": "Start anew",
    "button.use_refinement": "Use existing refinement",
    "button.retry": "Retry",
    "job.restart_upload_missing": (
        "Bot restarted and the upload for {file_name} is no "
        "longer available. Please send it again."
    ),
    "job.restart_resume_cache": (
        "Bot restarted; resuming {file_name} from cache."
    ),
    "job.restart_requeue": "Bot restarted; re-queuing {file_name}.",
    "job.restart_cache_review": (
        "Bot restarted; resuming the cache review for {file_name}."
    ),
    "job.restart_refine_review": (
        "Bot restarted; resuming the AI refinement review for "
        "{file_name}."
    ),
    "job.restart_send_refinement": (
        "Bot restarted; sending the completed AI refinement."
    ),
    "job.refinement_send_failed": (
        "AI refinement finished but could not be sent: {error}"
    ),
    "job.refine_failed_earlier": (
        "AI refinement for {file_name} failed earlier.\n\n"
        "Press Retry to run it again."
    ),
    "job.retry_unavailable": "This refinement can no longer be retried.",
    "job.retrying": "Retrying the AI refinement.",
    "job.action_text_processing": "Text processing",
    "job.action_transcription": "Transcription",
    "job.action_failed": "{action} failed for {file_name}: {error}",
    "job.text_upload_missing": (
        "The upload for {file_name} is no longer available. "
        "Please send it again."
    ),
    "job.reading": "Reading {file_name}...",
    "job.read_failed": (
        "Could not read {file_name} as UTF-8 text: {error}"
    ),
    "job.text_empty": (
        "Uploaded text for {file_name} is empty, "
        "so AI refinement cannot start."
    ),
    "job.invalid_srt": (
        "Uploaded SRT for {file_name} is not valid SRT."
    ),
    "job.using_cache": "Using the cached transcription for {file_name}.",
    "job.transcribing": "Transcribing {file_name}...",
    "job.raw_empty": (
        "Raw transcription for {file_name} is empty, "
        "so AI refinement cannot start."
    ),
    "job.refining": "Refining {file_name}...",
    "job.refine_failed": (
        "AI refinement failed: {error}\n\n"
        "Press Retry to run it again."
    ),
    "job.quota_exhausted": (
        "Gemini quota is exhausted. It should reset around {reset_at}.\n\n"
        "Press Retry after that."
    ),
    "job.model_unavailable": (
        "Gemini model is temporarily unavailable. It should recover around "
        "{reset_at}.\n\nPress Retry after that."
    ),
    "job.cache_existing_raw": (
        "Existing transcription found for {file_name}. "
        "Use it or start anew?"
    ),
    "job.cache_existing_refine": (
        "Existing AI refinement found for {file_name}. "
        "Use it or start a new refinement?"
    ),
    "job.raw_cache_inactive": "This cache choice is no longer active.",
    "job.re_transcribing": "Re-transcribing the file.",
    "job.using_existing_raw": "Using the existing transcription.",
    "job.refine_cache_inactive": (
        "This refinement cache choice is no longer active."
    ),
    "job.starting_new_refinement": "Starting a new AI refinement.",
    "job.cache_unreadable": (
        "The cache could not be read. Starting a new AI refinement."
    ),
    "job.using_existing_refinement": (
        "Using the existing AI refinement."
    ),
    "document.raw": "raw",
    "document.refined": "refined",
}

_ZH_HANS = {
    "auth.authenticated": (
        "您已通过验证。发送音频/视频文件或 .txt/.srt 文档即可开始。"
    ),
    "auth.locked": "尝试失败次数过多。请在约 {minutes} 分钟后重试。",
    "auth.question": (
        "{prefix}安全问题 {number}/{total}：\n\n"
        "{question}\n\n请回复您的答案。"
    ),
    "auth.correct_prefix": "正确。\n\n",
    "auth.incorrect_prefix": "错误。还剩 {attempts} 次尝试机会。\n\n",
    "auth.correct_authenticated": (
        "正确。您已通过验证。发送音频/视频文件或 .txt/.srt 文档即可开始。"
    ),
    "auth.status_required": "请先使用 /start 通过验证，再使用 /status。",
    "auth.upload_required": "请先使用 /start 通过验证，再上传文件。",
    "auth.callback_required": "请先使用 /start 通过验证，再使用此功能。",
    "status.none_admin": "没有进行中的任务。",
    "status.none_user": "您没有进行中的任务。",
    "status.header_admin": "所有进行中的任务（{count}）：",
    "status.header_user": "您进行中的任务（{count}）：",
    "status.line": (
        "{index}. {user_prefix}{file_name} - {status}"
        "（{lang}，{fmt}，任务 {job_id}）"
    ),
    "status.user_prefix": "[用户 {user_id}] ",
    "status.pending": "准备中",
    "status.queued": "排队中",
    "status.processing": "转录中",
    "status.raw_cache": "等待选择转录缓存",
    "status.refine_cache": "等待选择精修缓存",
    "status.refining": "精修中",
    "status.refine_failed": "精修失败（可重试）",
    "text.prompt_or_skip": "发送提示词，或按“跳过”。",
    "wizard.use_refinement": "使用 AI 精修？",
    "wizard.char_limit_invalid": (
        "请输入 10 到 100 之间的数字，作为每行字幕的目标字符数。"
    ),
    "wizard.use_buttons": "请使用上方的按钮。",
    "wizard.send_file": "发送音频/视频文件或 .txt/.srt 文档即可开始。",
    "upload.could_not_read": "无法读取该附件。",
    "upload.too_large": "该文件超过了配置的上传大小限制。",
    "upload.download_failed": "无法下载该文件：{error}",
    "upload.mixed": (
        "请勿在同一批次中混合上传媒体文件和 .txt/.srt 文件。"
        "请先完成或取消当前批次。"
    ),
    "upload.added": (
        "已将 {file_name} 添加到当前批次（共 {count} 个文件）。"
        "同一设置将应用于此批次中的每个文件。"
    ),
    "upload.received": (
        "已收到 {file_name}。\n\n"
        "您选择的设置将应用于本次向导完成前上传的每个文件。\n\n"
        "请选择语言。"
    ),
    "cache.unknown": "未知的缓存选项。",
    "cache.inactive": "该缓存选项已失效。",
    "cache.resolve_failed": "无法处理缓存选项：{error}",
    "cache.retry_failed": "无法重试精修：{error}",
    "cache.no_files": "没有等待设置的文件。请重新发送文件。",
    "confirm.inactive": "该确认已失效。",
    "confirm.edit_language": "请选择语言。此批次中已上传的文件将保留。",
    "confirm.cancelled": "已取消。已上传的文件已被删除。",
    "confirm.unknown": "未知的确认选项。",
    "language.unknown": "未知的语言选项。",
    "c2w.prompt": "将粤语口语转换为书面中文？",
    "char_limit.prompt": (
        "请输入每行字幕的目标字符数（10-100，例如 30）。"
    ),
    "format.language_chosen": "语言：{lang}\n\n请选择输出格式。",
    "format.choose_language_first": "请先选择语言。",
    "format.unknown": "未知的输出格式。",
    "prompt.whisper": "发送 faster-whisper 的初始提示词，或按“跳过”。",
    "zh.prompt": "请选择中文输出格式。",
    "zh.unknown": "未知的中文格式。",
    "selection.unknown": "未知的选项。",
    "manual.upload_prompt": (
        "手动 AI 精修。请发送要精修的 .txt 或 .srt 文件。您需要将每个"
        "提示词复制到您选择的 AI，并把回复发送回来。"
    ),
    "manual.text_only": "手动精修仅接受 .txt 或 .srt 文件。",
    "manual.single_file": (
        "手动精修一次只能处理一个文件。请先完成或取消当前精修。"
    ),
    "manual.invalid_response": "请以消息或 .txt/.srt 文件发送 AI 回复。",
    "manual.invalid_srt": "该回复不是有效的 SRT。请重新发送 AI 回复。",
    "manual.empty_response": "AI 回复为空。请重新发送。",
    "manual.read_failed": "无法读取回复文件：{error}",
    "manual.start_failed": "无法启动手动精修：{error}",
    "manual.prompt_header": (
        "手动精修提示词 {step}/{total}。请将该提示词复制到您的 AI，"
        "然后将完整回复以消息或 .txt/.srt 文件发送回来。"
    ),
    "manual.plan_header": (
        "手动结构规划提示词。请将该提示词复制到您的 AI，然后将完整 JSON "
        "回复以消息或 .txt 文件发送回来。"
    ),
    "manual.plan_applied": (
        "已接受结构规划。正在准备 {count} 个精修提示词。"
    ),
    "manual.plan_fallback": (
        "无法读取结构规划；正在使用自动分块。正在准备 {count} 个精修提示词。"
    ),
    "manual.exceed_unavailable": "当前没有等待回复的手动精修提示词。",
    "manual.exceed_impossible": (
        "该提示词无法进一步拆分。请尝试其他 AI，或发送您已有的回复。"
    ),
    "manual.exceed_split": (
        "当前提示词已拆分为 {count} 个较小的提示词。请回复下方的提示词。"
    ),
    "manual.merge_failed": "无法合并最终精修结果：{error}",
    "manual.send_failed": "无法发送手动精修结果：{error}",
    "manual.complete": "手动精修完成。最终精修文本已作为附件发送。",
    "manual.restart_resume": "机器人已重启；正在恢复手动精修。",
    "summary.review": "提交前请检查您的设置。",
    "summary.files": "文件数：{count}",
    "summary.language": "语言：{lang}",
    "summary.output_from_extension": (
        "输出格式：{formats}（根据文件扩展名）"
    ),
    "summary.output": "输出格式：{fmt}",
    "summary.zh_format": "中文格式：{zh}",
    "summary.initial_prompt": "初始提示词：{prompt}",
    "summary.none": "（无）",
    "summary.refine_required": "AI 精修：是（文本上传必需）",
    "summary.manual_refine": "AI 精修：手动（由您提供 AI 回复）",
    "summary.refine": "AI 精修：{value}",
    "summary.c2w": "粤语转书面中文：{value}",
    "summary.char_limit": "每行字幕字符数：{count}",
    "summary.footer": "确认、编辑设置，或取消。",
    "summary.yes": "是",
    "summary.no": "否",
    "finalize.missing": "部分设置缺失。请重新发送文件。",
    "finalize.submitted": "已提交 {count} 个文件。",
    "language.cantonese": "粤语",
    "language.mandarin": "普通话",
    "language.english": "英语",
    "format.text": "文本",
    "format.srt": "SRT",
    "zh.traditional": "繁体中文",
    "zh.simplified": "简体中文",
    "button.skip": "跳过",
    "button.yes": "是",
    "button.no": "否",
    "button.confirm": "确认",
    "button.edit": "编辑",
    "button.cancel": "取消",
    "button.use_transcription": "使用现有转录",
    "button.start_anew": "重新开始",
    "button.use_refinement": "使用现有精修",
    "button.retry": "重试",
    "job.restart_upload_missing": (
        "机器人已重启，{file_name} 的上传文件已不存在。请重新发送。"
    ),
    "job.restart_resume_cache": (
        "机器人已重启；正从缓存恢复 {file_name}。"
    ),
    "job.restart_requeue": "机器人已重启；正在重新排队 {file_name}。",
    "job.restart_cache_review": (
        "机器人已重启；正在恢复 {file_name} 的缓存选择。"
    ),
    "job.restart_refine_review": (
        "机器人已重启；正在恢复 {file_name} 的 AI 精修选择。"
    ),
    "job.restart_send_refinement": (
        "机器人已重启；正在发送已完成的 AI 精修。"
    ),
    "job.refinement_send_failed": (
        "AI 精修已完成，但发送失败：{error}"
    ),
    "job.refine_failed_earlier": (
        "{file_name} 的 AI 精修此前失败。\n\n按“重试”再次运行。"
    ),
    "job.retry_unavailable": "该精修已无法重试。",
    "job.retrying": "正在重试 AI 精修。",
    "job.action_text_processing": "文本处理",
    "job.action_transcription": "转录",
    "job.action_failed": "{file_name} 的{action}失败：{error}",
    "job.text_upload_missing": (
        "{file_name} 的上传文件已不存在。请重新发送。"
    ),
    "job.reading": "正在读取 {file_name}...",
    "job.read_failed": (
        "无法将 {file_name} 作为 UTF-8 文本读取：{error}"
    ),
    "job.text_empty": (
        "{file_name} 的上传文本为空，无法开始 AI 精修。"
    ),
    "job.invalid_srt": "{file_name} 的上传 SRT 不是有效的 SRT。",
    "job.using_cache": "正在使用 {file_name} 的缓存转录。",
    "job.transcribing": "正在转录 {file_name}...",
    "job.raw_empty": (
        "{file_name} 的原始转录为空，无法开始 AI 精修。"
    ),
    "job.refining": "正在精修 {file_name}...",
    "job.refine_failed": (
        "AI 精修失败：{error}\n\n按“重试”再次运行。"
    ),
    "job.quota_exhausted": (
        "Gemini 配额已用尽，预计将于 {reset_at} 重置。\n\n"
        "请在重置后按“重试”。"
    ),
    "job.model_unavailable": (
        "Gemini 模型暂时不可用，预计将于 {reset_at} 恢复。\n\n"
        "请在恢复后按“重试”。"
    ),
    "job.cache_existing_raw": (
        "找到 {file_name} 的现有转录。使用它还是重新开始？"
    ),
    "job.cache_existing_refine": (
        "找到 {file_name} 的现有 AI 精修。使用它还是开始新的精修？"
    ),
    "job.raw_cache_inactive": "该缓存选项已失效。",
    "job.re_transcribing": "正在重新转录该文件。",
    "job.using_existing_raw": "正在使用现有转录。",
    "job.refine_cache_inactive": "该精修缓存选项已失效。",
    "job.starting_new_refinement": "正在开始新的 AI 精修。",
    "job.cache_unreadable": "无法读取缓存。正在开始新的 AI 精修。",
    "job.using_existing_refinement": "正在使用现有 AI 精修。",
    "document.raw": "原始",
    "document.refined": "精修",
}

_ZH_HANT = {
    "auth.authenticated": (
        "您已通過驗證。傳送音訊/影片檔案或 .txt/.srt 文件即可開始。"
    ),
    "auth.locked": "嘗試失敗次數過多。請在約 {minutes} 分鐘後重試。",
    "auth.question": (
        "{prefix}安全性問題 {number}/{total}：\n\n"
        "{question}\n\n請回覆您的答案。"
    ),
    "auth.correct_prefix": "正確。\n\n",
    "auth.incorrect_prefix": "錯誤。還剩 {attempts} 次嘗試機會。\n\n",
    "auth.correct_authenticated": (
        "正確。您已通過驗證。傳送音訊/影片檔案或 .txt/.srt 文件即可開始。"
    ),
    "auth.status_required": "請先使用 /start 通過驗證，再使用 /status。",
    "auth.upload_required": "請先使用 /start 通過驗證，再上傳檔案。",
    "auth.callback_required": "請先使用 /start 通過驗證，再使用此功能。",
    "status.none_admin": "沒有進行中的任務。",
    "status.none_user": "您沒有進行中的任務。",
    "status.header_admin": "所有進行中的任務（{count}）：",
    "status.header_user": "您進行中的任務（{count}）：",
    "status.line": (
        "{index}. {user_prefix}{file_name} - {status}"
        "（{lang}，{fmt}，任務 {job_id}）"
    ),
    "status.user_prefix": "[使用者 {user_id}] ",
    "status.pending": "準備中",
    "status.queued": "排隊中",
    "status.processing": "轉錄中",
    "status.raw_cache": "等待選擇轉錄快取",
    "status.refine_cache": "等待選擇精修快取",
    "status.refining": "精修中",
    "status.refine_failed": "精修失敗（可重試）",
    "text.prompt_or_skip": "傳送提示詞，或按「跳過」。",
    "wizard.use_refinement": "使用 AI 精修？",
    "wizard.char_limit_invalid": (
        "請輸入 10 到 100 之間的數字，作為每行字幕的目標字元數。"
    ),
    "wizard.use_buttons": "請使用上方的按鈕。",
    "wizard.send_file": "傳送音訊/影片檔案或 .txt/.srt 文件即可開始。",
    "upload.could_not_read": "無法讀取該附件。",
    "upload.too_large": "該檔案超過了設定的上傳大小限制。",
    "upload.download_failed": "無法下載該檔案：{error}",
    "upload.mixed": (
        "請勿在同一批次中混合上傳媒體檔案和 .txt/.srt 檔案。"
        "請先完成或取消目前批次。"
    ),
    "upload.added": (
        "已將 {file_name} 加入目前批次（共 {count} 個檔案）。"
        "同一設定將套用於此批次中的每個檔案。"
    ),
    "upload.received": (
        "已收到 {file_name}。\n\n"
        "您選擇的設定將套用於本次精靈完成前上傳的每個檔案。\n\n"
        "請選擇語言。"
    ),
    "cache.unknown": "未知的快取選項。",
    "cache.inactive": "該快取選項已失效。",
    "cache.resolve_failed": "無法處理快取選項：{error}",
    "cache.retry_failed": "無法重試精修：{error}",
    "cache.no_files": "沒有等待設定的檔案。請重新傳送檔案。",
    "confirm.inactive": "該確認已失效。",
    "confirm.edit_language": "請選擇語言。此批次中已上傳的檔案將保留。",
    "confirm.cancelled": "已取消。已上傳的檔案已被刪除。",
    "confirm.unknown": "未知的確認選項。",
    "language.unknown": "未知的語言選項。",
    "c2w.prompt": "將粵語口語轉換為書面中文？",
    "char_limit.prompt": (
        "請輸入每行字幕的目標字元數（10-100，例如 30）。"
    ),
    "format.language_chosen": "語言：{lang}\n\n請選擇輸出格式。",
    "format.choose_language_first": "請先選擇語言。",
    "format.unknown": "未知的輸出格式。",
    "prompt.whisper": "傳送 faster-whisper 的初始提示詞，或按「跳過」。",
    "zh.prompt": "請選擇中文輸出格式。",
    "zh.unknown": "未知的中文格式。",
    "selection.unknown": "未知的選項。",
    "manual.upload_prompt": (
        "手動 AI 精修。請傳送要精修的 .txt 或 .srt 檔案。您需要將每個"
        "提示詞複製到您選擇的 AI，並把回覆傳送回來。"
    ),
    "manual.text_only": "手動精修僅接受 .txt 或 .srt 檔案。",
    "manual.single_file": (
        "手動精修一次只能處理一個檔案。請先完成或取消目前精修。"
    ),
    "manual.invalid_response": "請以訊息或 .txt/.srt 檔案傳送 AI 回覆。",
    "manual.invalid_srt": "該回覆不是有效的 SRT。請重新傳送 AI 回覆。",
    "manual.empty_response": "AI 回覆為空。請重新傳送。",
    "manual.read_failed": "無法讀取回覆檔案：{error}",
    "manual.start_failed": "無法啟動手動精修：{error}",
    "manual.prompt_header": (
        "手動精修提示詞 {step}/{total}。請將該提示詞複製到您的 AI，"
        "然後將完整回覆以訊息或 .txt/.srt 檔案傳送回來。"
    ),
    "manual.plan_header": (
        "手動結構規劃提示詞。請將該提示詞複製到您的 AI，然後將完整 JSON "
        "回覆以訊息或 .txt 檔案傳送回來。"
    ),
    "manual.plan_applied": (
        "已接受結構規劃。正在準備 {count} 個精修提示詞。"
    ),
    "manual.plan_fallback": (
        "無法讀取結構規劃；正在使用自動分塊。正在準備 {count} 個精修提示詞。"
    ),
    "manual.exceed_unavailable": "目前沒有等待回覆的手動精修提示詞。",
    "manual.exceed_impossible": (
        "該提示詞無法進一步拆分。請嘗試其他 AI，或傳送您已有的回覆。"
    ),
    "manual.exceed_split": (
        "目前提示詞已拆分為 {count} 個較小的提示詞。請回覆下方的提示詞。"
    ),
    "manual.merge_failed": "無法合併最終精修結果：{error}",
    "manual.send_failed": "無法傳送手動精修結果：{error}",
    "manual.complete": "手動精修完成。最終精修文字已作為附件傳送。",
    "manual.restart_resume": "機器人已重新啟動；正在恢復手動精修。",
    "summary.review": "提交前請檢查您的設定。",
    "summary.files": "檔案數：{count}",
    "summary.language": "語言：{lang}",
    "summary.output_from_extension": (
        "輸出格式：{formats}（根據副檔名）"
    ),
    "summary.output": "輸出格式：{fmt}",
    "summary.zh_format": "中文格式：{zh}",
    "summary.initial_prompt": "初始提示詞：{prompt}",
    "summary.none": "（無）",
    "summary.refine_required": "AI 精修：是（文字上傳必需）",
    "summary.manual_refine": "AI 精修：手動（由您提供 AI 回覆）",
    "summary.refine": "AI 精修：{value}",
    "summary.c2w": "粵語轉書面中文：{value}",
    "summary.char_limit": "每行字幕字元數：{count}",
    "summary.footer": "確認、編輯設定，或取消。",
    "summary.yes": "是",
    "summary.no": "否",
    "finalize.missing": "部分設定缺失。請重新傳送檔案。",
    "finalize.submitted": "已提交 {count} 個檔案。",
    "language.cantonese": "粵語",
    "language.mandarin": "普通話",
    "language.english": "英語",
    "format.text": "文字",
    "format.srt": "SRT",
    "zh.traditional": "繁體中文",
    "zh.simplified": "簡體中文",
    "button.skip": "跳過",
    "button.yes": "是",
    "button.no": "否",
    "button.confirm": "確認",
    "button.edit": "編輯",
    "button.cancel": "取消",
    "button.use_transcription": "使用現有轉錄",
    "button.start_anew": "重新開始",
    "button.use_refinement": "使用現有精修",
    "button.retry": "重試",
    "job.restart_upload_missing": (
        "機器人已重新啟動，{file_name} 的上傳檔案已不存在。請重新傳送。"
    ),
    "job.restart_resume_cache": (
        "機器人已重新啟動；正從快取恢復 {file_name}。"
    ),
    "job.restart_requeue": "機器人已重新啟動；正在重新排隊 {file_name}。",
    "job.restart_cache_review": (
        "機器人已重新啟動；正在恢復 {file_name} 的快取選擇。"
    ),
    "job.restart_refine_review": (
        "機器人已重新啟動；正在恢復 {file_name} 的 AI 精修選擇。"
    ),
    "job.restart_send_refinement": (
        "機器人已重新啟動；正在傳送已完成的 AI 精修。"
    ),
    "job.refinement_send_failed": (
        "AI 精修已完成，但傳送失敗：{error}"
    ),
    "job.refine_failed_earlier": (
        "{file_name} 的 AI 精修先前失敗。\n\n按「重試」再次執行。"
    ),
    "job.retry_unavailable": "該精修已無法重試。",
    "job.retrying": "正在重試 AI 精修。",
    "job.action_text_processing": "文字處理",
    "job.action_transcription": "轉錄",
    "job.action_failed": "{file_name} 的{action}失敗：{error}",
    "job.text_upload_missing": (
        "{file_name} 的上傳檔案已不存在。請重新傳送。"
    ),
    "job.reading": "正在讀取 {file_name}...",
    "job.read_failed": (
        "無法將 {file_name} 作為 UTF-8 文字讀取：{error}"
    ),
    "job.text_empty": (
        "{file_name} 的上傳文字為空，無法開始 AI 精修。"
    ),
    "job.invalid_srt": "{file_name} 的上傳 SRT 不是有效的 SRT。",
    "job.using_cache": "正在使用 {file_name} 的快取轉錄。",
    "job.transcribing": "正在轉錄 {file_name}...",
    "job.raw_empty": (
        "{file_name} 的原始轉錄為空，無法開始 AI 精修。"
    ),
    "job.refining": "正在精修 {file_name}...",
    "job.refine_failed": (
        "AI 精修失敗：{error}\n\n按「重試」再次執行。"
    ),
    "job.quota_exhausted": (
        "Gemini 配額已用盡，預計將於 {reset_at} 重置。\n\n"
        "請在重置後按「重試」。"
    ),
    "job.model_unavailable": (
        "Gemini 模型暫時不可用，預計將於 {reset_at} 恢復。\n\n"
        "請在恢復後按「重試」。"
    ),
    "job.cache_existing_raw": (
        "找到 {file_name} 的現有轉錄。要使用它還是重新開始？"
    ),
    "job.cache_existing_refine": (
        "找到 {file_name} 的現有 AI 精修。要使用它還是開始新的精修？"
    ),
    "job.raw_cache_inactive": "該快取選項已失效。",
    "job.re_transcribing": "正在重新轉錄該檔案。",
    "job.using_existing_raw": "正在使用現有轉錄。",
    "job.refine_cache_inactive": "該精修快取選項已失效。",
    "job.starting_new_refinement": "正在開始新的 AI 精修。",
    "job.cache_unreadable": "無法讀取快取。正在開始新的 AI 精修。",
    "job.using_existing_refinement": "正在使用現有 AI 精修。",
    "document.raw": "原始",
    "document.refined": "精修",
}

_CATALOGS = {
    "en": _EN,
    "zh-Hans": _ZH_HANS,
    "zh-Hant": _ZH_HANT,
}


def resolve_locale(language_code: Optional[str]) -> str:
    """Maps a Telegram ``language_code`` to one of the supported locales."""
    code = (language_code or "").strip().lower()
    if not code:
        return DEFAULT_LOCALE
    if code.startswith("zh-hant") or code in ("zh-hk", "zh-tw", "zh-mo"):
        return "zh-Hant"
    if (
        code == "zh"
        or code.startswith("zh-hans")
        or code in ("zh-cn", "zh-sg")
    ):
        return "zh-Hans"
    return DEFAULT_LOCALE


def locale_for_update(update) -> str:
    """Resolves the sender's locale from a Telegram update."""
    user = getattr(update, "effective_user", None)
    return resolve_locale(getattr(user, "language_code", None))


def t(locale: str, key: str, **kwargs: Any) -> str:
    """Returns a localized message, falling back to English then the key."""
    catalog = _CATALOGS.get(locale)
    template: Optional[str] = None
    if catalog is not None:
        template = catalog.get(key)
    if template is None:
        template = _EN.get(key, key)
    return template.format(**kwargs) if kwargs else template


def _display(locale: str, value: Optional[str], keys: Dict[str, str]) -> str:
    if not value:
        return value or ""
    key = keys.get(value)
    return t(locale, key) if key else value


def display_language(locale: str, value: Optional[str]) -> str:
    return _display(locale, value, _LANGUAGE_KEYS)


def display_format(locale: str, value: Optional[str]) -> str:
    return _display(locale, value, _FORMAT_KEYS)


def display_zh_format(locale: str, value: Optional[str]) -> str:
    return _display(locale, value, _ZH_KEYS)
