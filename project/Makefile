audit:
	python app/tools/phase2_label_audit.py

batch:
	python app/tools/phase2_batch_runner.py

draft:
	python app/tools/phase2_expected_draft.py

phase2-all:
	python app/tools/phase2_label_audit.py
	python app/tools/phase2_batch_runner.py
	python app/tools/phase2_expected_draft.py

story:
	python app/services/scene_story_service.py --audio-id 478db4fea37e

phase3-test:
	python app/services/scene_story_service.py --audio-id 478db4fea37e
	cat data/story/478db4fea37e/story.json
