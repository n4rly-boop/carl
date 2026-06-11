"""
Tests for AgentSkillStep feature.

Covers:
  - AgentSkillSource coercion from plain strings
  - SkillLoader loading from explicit path
  - SkillLoader loading by name
  - SkillLoader: SkillNotFoundError for unknown names
  - SKILL.md parsing (frontmatter + body)
  - Resource discovery (scripts/, references/, assets/)
  - AgentSkillStepExecutor in LLM mode
  - AgentSkillStepExecutor in SCRIPT mode
  - AgentSkillStepExecutor script timeout
  - AgentSkillStepDescription serialization (model_dump includes step_type)
"""

import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_carl.models.agent_skill import (
    AgentSkillExecutionMode,
    AgentSkillSource,
    AgentSkillStepConfig,
)
from mmar_carl.models.steps import AgentSkillStepDescription
from mmar_carl.models.enums import StepType
from mmar_carl.skill_loader import (
    SkillLoader,
    SkillNotFoundError,
    SkillParseError,
    _parse_skill_md,
    _build_manifest,
)
from mmar_carl.step_executors import AgentSkillStepExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_skill_dir(tmp_path: Path, name: str, description: str, instructions: str = "Do stuff.") -> Path:
    """Create a minimal skill directory with a SKILL.md file."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    # Use explicit string concatenation to avoid indentation issues with textwrap.dedent
    content = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "license: MIT\n"
        "---\n"
        f"{instructions}\n"
    )
    skill_md.write_text(content)
    return skill_dir


def make_skill_md_file(path: Path, name: str, description: str, extra_frontmatter: str = "", body: str = "Instructions here.") -> None:
    """Write a SKILL.md to the given path."""
    fm_extra = f"\n{extra_frontmatter}" if extra_frontmatter else ""
    content = textwrap.dedent(f"""\
        ---
        name: {name}
        description: {description}{fm_extra}
        ---
        {body}
    """)
    path.write_text(content)


def make_mock_context(history: list[str] | None = None) -> MagicMock:
    """Create a minimal mock ReasoningContext."""
    ctx = MagicMock()
    ctx.history = history or []
    ctx.language = "en"
    ctx.retry_max = 1
    ctx.metadata = {}
    # MagicMock's default makes is_cancelled() return a truthy MagicMock;
    # intra-step polls would short-circuit. Force False.
    ctx.is_cancelled = MagicMock(return_value=False)

    def memory_write(key, value, namespace="default"):
        ctx.metadata[f"_mem_{namespace}_{key}"] = value

    def memory_read(key, namespace="default", default=None):
        return ctx.metadata.get(f"_mem_{namespace}_{key}", default)

    ctx.memory_write.side_effect = memory_write
    ctx.memory_read.side_effect = memory_read
    return ctx


# ---------------------------------------------------------------------------
# Test 1: AgentSkillSource string coercion
# ---------------------------------------------------------------------------

class TestAgentSkillSourceCoercion:
    def test_plain_name_coerced_to_name(self):
        config = AgentSkillStepConfig(skill="pdf", task="do something")
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.name == "pdf"
        assert config.skill.path is None
        assert config.skill.git_url is None

    def test_absolute_path_coerced_to_path(self):
        config = AgentSkillStepConfig(skill="/usr/local/skills/pdf", task="do something")
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.path == "/usr/local/skills/pdf"
        assert config.skill.name is None

    def test_relative_path_coerced_to_path(self):
        config = AgentSkillStepConfig(skill="./local/skills/pdf", task="do something")
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.path == "./local/skills/pdf"

    def test_trailing_slash_coerced_to_path(self):
        config = AgentSkillStepConfig(skill="some/dir/", task="do something")
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.path == "some/dir/"

    def test_https_url_coerced_to_git_url(self):
        config = AgentSkillStepConfig(skill="https://github.com/org/skills", task="do something")
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.git_url == "https://github.com/org/skills"
        assert config.skill.name is None

    def test_git_at_url_coerced_to_git_url(self):
        config = AgentSkillStepConfig(skill="git@github.com:org/skills.git", task="do something")
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.git_url == "git@github.com:org/skills.git"

    def test_direct_source_object_passthrough(self):
        source = AgentSkillSource(name="pptx")
        config = AgentSkillStepConfig(skill=source, task="create slides")
        assert config.skill is source


# ---------------------------------------------------------------------------
# Test 2: SkillLoader.load() from explicit path
# ---------------------------------------------------------------------------

class TestSkillLoaderFromPath:
    def test_load_from_path(self, tmp_path):
        skill_dir = make_skill_dir(tmp_path, "pdf", "Extract content from PDF files")
        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=False)
        manifest = loader.load_sync(source)

        assert manifest.name == "pdf"
        assert manifest.description == "Extract content from PDF files"
        assert manifest.license == "MIT"
        assert "Do stuff." in manifest.instructions
        assert str(skill_dir.resolve()) == manifest.skill_dir

    def test_load_from_path_missing_skill_md(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        source = AgentSkillSource(path=str(empty_dir))
        loader = SkillLoader(enable_cache=False)
        with pytest.raises(SkillNotFoundError):
            loader.load_sync(source)

    def test_load_from_path_tilde_expansion(self, tmp_path, monkeypatch):
        skill_dir = make_skill_dir(tmp_path, "pptx", "Create PPTX presentations")
        monkeypatch.setenv("HOME", str(tmp_path))
        # Make a skills dir under "home"
        home_skills = tmp_path / "agents" / "skills" / "pptx"
        home_skills.mkdir(parents=True)
        (home_skills / "SKILL.md").write_text(
            "---\nname: pptx\ndescription: PPTX desc\n---\nBody.\n"
        )
        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=False)
        manifest = loader.load_sync(source)
        assert manifest.name == "pptx"


# ---------------------------------------------------------------------------
# Test 3: SkillLoader.load() by name
# ---------------------------------------------------------------------------

class TestSkillLoaderByName:
    def test_load_by_name_found(self, tmp_path):
        search_root = tmp_path / "skills"
        search_root.mkdir()
        skill_dir = search_root / "pdf"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: pdf\ndescription: PDF skill\n---\nInstructions.\n"
        )
        source = AgentSkillSource(name="pdf", search_paths=[str(search_root)])
        loader = SkillLoader(enable_cache=False)
        manifest = loader.load_sync(source)
        assert manifest.name == "pdf"

    def test_load_by_name_not_found(self):
        source = AgentSkillSource(name="nonexistent_skill_xyzzy")
        loader = SkillLoader(enable_cache=False)
        with pytest.raises(SkillNotFoundError, match="nonexistent_skill_xyzzy"):
            loader.load_sync(source)


# ---------------------------------------------------------------------------
# Test 4: SkillNotFoundError for unknown name
# ---------------------------------------------------------------------------

class TestSkillNotFound:
    def test_raises_skill_not_found_error(self):
        loader = SkillLoader(enable_cache=False)
        with pytest.raises(SkillNotFoundError):
            loader.load_sync(AgentSkillSource(name="definitely_not_a_real_skill"))

    def test_error_message_contains_skill_name(self):
        loader = SkillLoader(enable_cache=False)
        try:
            loader.load_sync(AgentSkillSource(name="my_missing_skill"))
        except SkillNotFoundError as e:
            assert "my_missing_skill" in str(e)


# ---------------------------------------------------------------------------
# Test 5: SKILL.md parsing
# ---------------------------------------------------------------------------

class TestSkillMdParsing:
    def test_basic_frontmatter_and_body(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(textwrap.dedent("""\
            ---
            name: myskill
            description: A test skill
            license: Apache-2.0
            ---
            # Guide
            Do the thing.
        """))
        result = _parse_skill_md(str(skill_md))
        fm = result["frontmatter"]
        assert fm["name"] == "myskill"
        assert fm["description"] == "A test skill"
        assert fm["license"] == "Apache-2.0"
        assert "# Guide" in result["instructions"]
        assert "Do the thing." in result["instructions"]

    def test_lenient_yaml_unquoted_colon_in_value(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(textwrap.dedent("""\
            ---
            name: colon-test
            description: Use this skill: it's great
            ---
            Body text.
        """))
        result = _parse_skill_md(str(skill_md))
        # description should contain the colon
        assert ":" in result["frontmatter"]["description"]

    def test_missing_opening_delimiter_raises(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("name: foo\ndescription: bar\n")
        with pytest.raises(SkillParseError):
            _parse_skill_md(str(skill_md))

    def test_missing_closing_delimiter_raises(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\ndescription: bar\n")
        with pytest.raises(SkillParseError):
            _parse_skill_md(str(skill_md))

    def test_missing_name_field_raises(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\ndescription: No name here\n---\nBody.\n")
        with pytest.raises(SkillParseError, match="name"):
            _build_manifest(str(skill_dir), str(skill_md))

    def test_missing_description_field_raises(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: skill_without_desc\n---\nBody.\n")
        with pytest.raises(SkillParseError, match="description"):
            _build_manifest(str(skill_dir), str(skill_md))

    def test_extra_frontmatter_fields_in_metadata(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(textwrap.dedent("""\
            ---
            name: ext
            description: Extended
            version: 1.2.3
            author: Test
            ---
            Body.
        """))
        manifest = _build_manifest(str(skill_dir), str(skill_md))
        assert manifest.metadata.get("version") == "1.2.3"
        assert manifest.metadata.get("author") == "Test"


# ---------------------------------------------------------------------------
# Test 6: Resource discovery (scripts, references, assets)
# ---------------------------------------------------------------------------

class TestSkillResourceDiscovery:
    def test_discovers_scripts_and_references(self, tmp_path):
        skill_dir = tmp_path / "skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: res\ndescription: Resource test\n---\nDo stuff.\n"
        )
        # Create resource files
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "extract.py").write_text("print('hello')")
        (skill_dir / "references").mkdir()
        (skill_dir / "references" / "REFERENCE.md").write_text("# Ref")
        (skill_dir / "assets").mkdir()
        (skill_dir / "assets" / "template.pptx").write_bytes(b"\x00")

        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=False)
        manifest = loader.load_sync(source)

        assert any("extract.py" in s for s in manifest.scripts)
        assert any("REFERENCE.md" in r for r in manifest.references)
        assert any("template.pptx" in a for a in manifest.assets)

    def test_empty_skill_has_no_resources(self, tmp_path):
        skill_dir = make_skill_dir(tmp_path, "empty_skill", "No resources")
        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=False)
        manifest = loader.load_sync(source)
        assert manifest.scripts == []
        assert manifest.references == []
        assert manifest.assets == []


# ---------------------------------------------------------------------------
# Test 7: AgentSkillStepExecutor — LLM mode
# ---------------------------------------------------------------------------

class TestAgentSkillStepLLMMode:
    @pytest.mark.asyncio
    async def test_llm_mode_uses_skill_instructions_in_prompt(self, tmp_path):
        """The skill instructions should appear in the system prompt sent to LLM."""
        instructions = "## PDF Processing\nAlways extract all tables and figures."
        skill_dir = make_skill_dir(tmp_path, "pdf", "PDF extraction skill", instructions=instructions)

        config = AgentSkillStepConfig(
            skill=str(skill_dir) + "/",  # path-style
            task="Extract text from the file at {pdf_path}",
            input_mapping={"pdf_path": "$memory.input.pdf_path"},
            execution_mode=AgentSkillExecutionMode.LLM,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Read PDF"
        step.step_config = config

        context = make_mock_context()
        context.metadata["_mem_input_pdf_path"] = "/tmp/test.pdf"
        context.memory_read.side_effect = lambda key, namespace="default", default=None: (
            "/tmp/test.pdf" if key == "pdf_path" and namespace == "input" else default
        )

        # Mock LLM client
        mock_llm = AsyncMock()
        mock_llm.get_response_with_system = AsyncMock(return_value="Extracted text content")
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert result.result == "Extracted text content"

        # Verify LLM was called and instructions appeared in the system prompt
        call_args = mock_llm.get_response_with_system.call_args
        system_prompt = call_args[1].get("system_prompt") or call_args[0][0]
        assert "PDF Processing" in system_prompt or "pdf" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_llm_mode_interpolates_input_mapping(self, tmp_path):
        """Input mapping values should be substituted into the task string."""
        skill_dir = make_skill_dir(tmp_path, "pdf", "PDF skill", instructions="Process PDFs.")

        captured_user_prompts: list[str] = []

        async def capture_prompt(system_prompt, user_prompt, retries=3):
            captured_user_prompts.append(user_prompt)
            return "Done"

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Process the file at {file_path} with mode {mode}",
            input_mapping={
                "file_path": "$memory.input.path",
                "mode": "$memory.input.mode",
            },
            execution_mode=AgentSkillExecutionMode.LLM,
        )

        step = MagicMock()
        step.number = 2
        step.title = "Process"
        step.step_config = config

        context = make_mock_context()

        def mock_memory_read(key, namespace="default", default=None):
            data = {"path": "/data/file.pdf", "mode": "fast"}
            return data.get(key, default)

        context.memory_read.side_effect = mock_memory_read

        mock_llm = AsyncMock()
        mock_llm.get_response_with_system = AsyncMock(side_effect=capture_prompt)
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert len(captured_user_prompts) == 1
        prompt = captured_user_prompts[0]
        assert "/data/file.pdf" in prompt
        assert "fast" in prompt

    @pytest.mark.asyncio
    async def test_llm_mode_history_entry_format(self, tmp_path):
        """History entry should follow 'Step N. Title [skill:name]\\nResult: ...' format."""
        skill_dir = make_skill_dir(tmp_path, "testskill", "A test skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Do a thing",
            execution_mode=AgentSkillExecutionMode.LLM,
        )

        step = MagicMock()
        step.number = 3
        step.title = "My Step"
        step.step_config = config

        context = make_mock_context(history=["Previous step result"])

        mock_llm = AsyncMock()
        mock_llm.get_response_with_system = AsyncMock(return_value="Skill output text")
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert len(result.updated_history) == 2
        entry = result.updated_history[1]
        assert "Step 3." in entry
        assert "My Step" in entry
        assert "[skill:testskill]" in entry
        assert "Skill output text" in entry


# ---------------------------------------------------------------------------
# Test 8: AgentSkillStepExecutor — SCRIPT mode
# ---------------------------------------------------------------------------

class TestAgentSkillStepScriptMode:
    @pytest.mark.asyncio
    async def test_script_mode_captures_stdout(self, tmp_path):
        """Script mode should capture stdout as the result."""
        skill_dir = tmp_path / "scriptskill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: scriptskill\ndescription: A script skill\n---\nRun scripts.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / "run.py"
        script_file.write_text("print('extracted text output')\n")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Extract text",
            execution_mode=AgentSkillExecutionMode.SCRIPT,
            python_executable=sys.executable,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Run Script"
        step.step_config = config

        context = make_mock_context()
        context.get_llm_client_for_step = MagicMock()

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert "extracted text output" in result.result

    @pytest.mark.asyncio
    async def test_script_mode_nonzero_exit_returns_failure(self, tmp_path):
        """Non-zero exit code should produce success=False result."""
        skill_dir = tmp_path / "failskill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: failskill\ndescription: Fails on purpose\n---\nFail.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / "fail.py"
        script_file.write_text("import sys; sys.stderr.write('error detail'); sys.exit(1)\n")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Do something",
            execution_mode=AgentSkillExecutionMode.SCRIPT,
            python_executable=sys.executable,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Fail Script"
        step.step_config = config

        context = make_mock_context()
        context.get_llm_client_for_step = MagicMock()

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is False
        assert "exit" in result.error_message.lower() or "1" in result.error_message


# ---------------------------------------------------------------------------
# Test 9: Script timeout
# ---------------------------------------------------------------------------

class TestAgentSkillStepScriptTimeout:
    @pytest.mark.asyncio
    async def test_script_timeout_returns_failure(self, tmp_path):
        """A script that sleeps too long should time out and return success=False."""
        skill_dir = tmp_path / "slowskill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: slowskill\ndescription: Sleeps forever\n---\nSlow.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / "slow.py"
        script_file.write_text("import time; time.sleep(60)\n")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Do slow thing",
            execution_mode=AgentSkillExecutionMode.SCRIPT,
            python_executable=sys.executable,
            timeout=0.5,  # Very short timeout
        )

        step = MagicMock()
        step.number = 1
        step.title = "Slow Script"
        step.step_config = config

        context = make_mock_context()
        context.get_llm_client_for_step = MagicMock()

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is False
        assert "timeout" in result.error_message.lower() or "timed out" in result.error_message.lower()


# ---------------------------------------------------------------------------
# Test 10: skill not found returns graceful failure
# ---------------------------------------------------------------------------

class TestAgentSkillNotFoundGraceful:
    @pytest.mark.asyncio
    async def test_not_found_returns_success_false(self):
        """SkillNotFoundError should be caught and returned as success=False."""
        config = AgentSkillStepConfig(
            skill="totally_nonexistent_skill_xyzzy_12345",
            task="Do something",
        )

        step = MagicMock()
        step.number = 1
        step.title = "Missing Skill"
        step.step_config = config

        context = make_mock_context()
        context.get_llm_client_for_step = MagicMock()

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is False
        assert result.error_message


# ---------------------------------------------------------------------------
# Test 11: AgentSkillStepDescription serialization
# ---------------------------------------------------------------------------

class TestAgentSkillStepDescriptionSerialization:
    def test_model_dump_includes_step_type(self):
        config = AgentSkillStepConfig(skill="pdf", task="Extract content")
        desc = AgentSkillStepDescription(number=1, title="Read PDF", config=config)
        data = desc.model_dump()
        assert data["step_type"] == "agent_skill"
        assert data["number"] == 1
        assert data["title"] == "Read PDF"

    def test_step_type_property(self):
        config = AgentSkillStepConfig(skill="pdf", task="Extract content")
        desc = AgentSkillStepDescription(number=1, title="Read PDF", config=config)
        assert desc.step_type == StepType.AGENT_SKILL
        assert desc.step_type == "agent_skill"

    def test_step_config_property(self):
        config = AgentSkillStepConfig(skill="pdf", task="Extract content")
        desc = AgentSkillStepDescription(number=1, title="Read PDF", config=config)
        assert desc.step_config is config


# ---------------------------------------------------------------------------
# Test 12: SkillLoader in-memory caching
# ---------------------------------------------------------------------------

class TestSkillLoaderCaching:
    def test_cached_manifest_returned_on_second_load(self, tmp_path):
        skill_dir = make_skill_dir(tmp_path, "cached_skill", "Test caching")
        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=True)

        manifest1 = loader.load_sync(source)
        manifest2 = loader.load_sync(source)

        # Same object (cache hit)
        assert manifest1 is manifest2

    def test_clear_cache(self, tmp_path):
        skill_dir = make_skill_dir(tmp_path, "cache_clear_skill", "Test cache clear")
        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=True)

        manifest1 = loader.load_sync(source)
        loader.clear_cache()
        manifest2 = loader.load_sync(source)

        # Different objects after cache clear
        assert manifest1 is not manifest2
        # But same content
        assert manifest1.name == manifest2.name


# ---------------------------------------------------------------------------
# Test 13: AgentSkillSource validation
# ---------------------------------------------------------------------------

class TestAgentSkillSourceValidation:
    def test_must_have_exactly_one_source(self):
        with pytest.raises(Exception):
            AgentSkillSource(name="foo", path="/bar")

    def test_all_none_raises(self):
        with pytest.raises(Exception):
            AgentSkillSource()


# ---------------------------------------------------------------------------
# Test 14: Chain serialization round-trip
# ---------------------------------------------------------------------------

class TestChainSerializationRoundTrip:
    """Verify AgentSkillStep survives JSON serialization and deserialization via chain."""

    def test_chain_to_json_and_back(self, tmp_path):
        """Chain with AgentSkillStepDescription should round-trip through JSON."""
        import json as _json
        from mmar_carl.chain import ReasoningChain
        from mmar_carl.models.steps import LLMStepDescription

        config = AgentSkillStepConfig(
            skill="pdf",
            task="Extract text from {pdf_path}",
            input_mapping={"pdf_path": "$memory.input.pdf_path"},
            timeout=60.0,
        )
        chain = ReasoningChain(
            steps=[
                AgentSkillStepDescription(number=1, title="Read PDF", config=config),
                LLMStepDescription(
                    number=2,
                    title="Analyze",
                    dependencies=[1],
                    aim="Summarize the extracted text",
                ),
            ]
        )

        # Serialize
        json_str = chain.to_json()
        data = _json.loads(json_str)
        assert any(s.get("step_type") == "agent_skill" for s in data["steps"])

        # Deserialize using from_dict_typed
        restored = ReasoningChain.from_dict_typed(data)
        assert len(restored.steps) == 2
        skill_step = restored.steps[0]
        assert skill_step.step_type == StepType.AGENT_SKILL
        assert skill_step.step_config.task == "Extract text from {pdf_path}"
        assert skill_step.step_config.timeout == 60.0

    def test_agent_skill_source_roundtrip(self):
        """AgentSkillSource should serialize and deserialize correctly."""
        config = AgentSkillStepConfig(
            skill=AgentSkillSource(name="pptx", search_paths=["/custom/path"]),
            task="Create slides",
        )
        desc = AgentSkillStepDescription(number=1, title="Make slides", config=config)
        data = desc.model_dump()
        # The skill field should be a dict (serialized AgentSkillSource)
        assert isinstance(data["config"]["skill"], dict)
        assert data["config"]["skill"]["name"] == "pptx"


# ---------------------------------------------------------------------------
# Test 15: AgentSkillStep integrated in a full chain execution
# ---------------------------------------------------------------------------

class TestAgentSkillStepInChain:
    """Verify AgentSkillStep works as a real step within a DAGExecutor chain."""

    @pytest.mark.asyncio
    async def test_skill_step_result_in_history(self, tmp_path):
        """AgentSkillStep result should appear in history for downstream LLM steps."""
        from mmar_carl.chain import ReasoningChain
        from mmar_carl.models.context import ReasoningContext
        from mmar_carl.models.steps import LLMStepDescription

        skill_dir = make_skill_dir(
            tmp_path, "mypdf", "PDF skill for tests",
            instructions="Process PDF files."
        )

        # Use the mock client pattern from other tests
        mock_api = MagicMock()
        mock_api.get_response_with_retries = AsyncMock(return_value="LLM response about the document")
        mock_api.get_response = AsyncMock(return_value="LLM response about the document")
        mock_api.get_response_with_system = AsyncMock(return_value="LLM response about the document")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Extract content from the document",
        )

        chain = ReasoningChain(
            steps=[
                AgentSkillStepDescription(number=1, title="Read with skill", config=config),
                LLMStepDescription(
                    number=2,
                    title="Analyze",
                    dependencies=[1],
                    aim="Summarize extracted content",
                ),
            ]
        )

        context = ReasoningContext(
            outer_context="Test document",
            api=mock_api,
            model="test-model",
        )

        result = await chain.execute_async(context)

        assert result.success is True
        assert len(result.step_results) == 2
        # History should contain an entry from the skill step
        combined_history = "\n".join(context.history)
        assert "skill:mypdf" in combined_history


# ---------------------------------------------------------------------------
# Test 16: SUBAGENT execution mode
# ---------------------------------------------------------------------------

class TestAgentSkillStepSubagentMode:
    """SUBAGENT mode: script data is injected into LLM user message."""

    @pytest.mark.asyncio
    async def test_subagent_passes_script_output_to_llm(self, tmp_path):
        """When script succeeds, its stdout should appear in the LLM user message."""
        skill_dir = tmp_path / "pdfskill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: pdfskill\ndescription: PDF skill\n---\nProcess PDFs.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "extract.py").write_text(
            "print('## Page 1\\n\\nHello World')\n"
        )

        captured_user_prompts: list[str] = []

        async def capture(system_prompt, user_prompt, retries=3):
            captured_user_prompts.append(user_prompt)
            return "Interpreted: Hello World document"

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Summarize the document at {pdf_path}",
            input_mapping={"pdf_path": "/tmp/test.pdf"},
            execution_mode=AgentSkillExecutionMode.SUBAGENT,
            script_name="scripts/extract.py",
            python_executable=sys.executable,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Subagent PDF"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_system = AsyncMock(side_effect=capture)
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert "Interpreted" in result.result
        # Script output should be injected into the user message
        assert len(captured_user_prompts) == 1
        user_msg = captured_user_prompts[0]
        assert "Hello World" in user_msg
        assert "Data from skill script" in user_msg

    @pytest.mark.asyncio
    async def test_subagent_falls_back_to_llm_when_script_fails(self, tmp_path):
        """When script fails, SUBAGENT should still call LLM with a fallback note."""
        skill_dir = tmp_path / "badscript"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: badscript\ndescription: A skill\n---\nInstructions.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "fail.py").write_text(
            "import sys; sys.exit(1)\n"
        )

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Do the thing",
            execution_mode=AgentSkillExecutionMode.SUBAGENT,
            script_name="scripts/fail.py",
            python_executable=sys.executable,
        )

        step = MagicMock()
        step.number = 2
        step.title = "Subagent Fallback"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_system = AsyncMock(return_value="LLM fallback result")
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        # Should still succeed (LLM fallback)
        assert result.success is True
        assert result.result == "LLM fallback result"

    @pytest.mark.asyncio
    async def test_subagent_metadata_records_script_usage(self, tmp_path):
        """SUBAGENT metadata should record whether script was used."""
        skill_dir = tmp_path / "metaskill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: metaskill\ndescription: Meta skill\n---\nDo things.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "run.py").write_text("print('script output')\n")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Run the task",
            execution_mode=AgentSkillExecutionMode.SUBAGENT,
            script_name="scripts/run.py",
            python_executable=sys.executable,
        )

        step = MagicMock()
        step.number = 3
        step.title = "Meta Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_system = AsyncMock(return_value="LLM result")
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert result.result_data.get("script_used") is True
        assert result.result_data.get("execution_mode") == "subagent"


# ---------------------------------------------------------------------------
# Test 16: URI coercion — new URI schemes
# ---------------------------------------------------------------------------

class TestURICoercion:
    """Tests for new URI scheme coercion in AgentSkillStepConfig."""

    def test_github_uri_coerced_to_git_url(self):
        """github:// URIs should map to git_url + git_ref + git_subdirectory."""
        config = AgentSkillStepConfig(
            skill="github://anthropics/skills/skills/pdf@main",
            task="Extract text",
        )
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.git_url == "https://github.com/anthropics/skills"
        assert config.skill.git_ref == "main"
        assert config.skill.git_subdirectory == "skills/pdf"

    def test_github_uri_default_ref_is_main(self):
        config = AgentSkillStepConfig(
            skill="github://anthropics/skills/skills/pptx",
            task="Create deck",
        )
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.git_ref == "main"

    def test_github_uri_no_subpath(self):
        """github://owner/repo with no subpath should set git_subdirectory=None."""
        config = AgentSkillStepConfig(
            skill="github://myorg/my-skill",
            task="Do something",
        )
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.git_url == "https://github.com/myorg/my-skill"
        assert config.skill.git_subdirectory is None

    def test_module_uri_coerced_to_package(self):
        config = AgentSkillStepConfig(
            skill="module://my_pkg.skills.pdf",
            task="Do something",
        )
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.package == "my_pkg.skills.pdf"
        assert config.skill.git_url is None

    def test_local_uri_coerced_to_path(self):
        config = AgentSkillStepConfig(
            skill="local:///home/user/skills/my-skill",
            task="Do something",
        )
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.path == "/home/user/skills/my-skill"

    def test_github_uri_custom_ref(self):
        config = AgentSkillStepConfig(
            skill="github://anthropics/skills/skills/pdf@v2.0",
            task="Do something",
        )
        assert isinstance(config.skill, AgentSkillSource)
        assert config.skill.git_ref == "v2.0"
        assert config.skill.git_subdirectory == "skills/pdf"


# ---------------------------------------------------------------------------
# Test 17: New config fields
# ---------------------------------------------------------------------------

class TestNewConfigFields:
    """Tests for new AgentSkillStepConfig fields (v0.3 additions)."""

    def test_default_trust_policy(self):
        config = AgentSkillStepConfig(skill="pdf", task="do")
        assert config.trust_policy == "any"

    def test_default_llm_max_iterations(self):
        config = AgentSkillStepConfig(skill="pdf", task="do")
        assert config.llm_max_iterations == 8

    def test_default_output_capture(self):
        config = AgentSkillStepConfig(skill="pdf", task="do")
        assert config.output_capture == "both"

    def test_default_output_files_glob(self):
        config = AgentSkillStepConfig(skill="pdf", task="do")
        assert config.output_files_glob == ["*"]

    def test_default_runtime(self):
        config = AgentSkillStepConfig(skill="pdf", task="do")
        assert config.runtime == "local"

    def test_skill_sha256_field(self):
        config = AgentSkillStepConfig(
            skill="pdf",
            task="do",
            skill_sha256="abc123",
            trust_policy="sha_pinned",
        )
        assert config.skill_sha256 == "abc123"
        assert config.trust_policy == "sha_pinned"

    def test_extra_pip_field(self):
        config = AgentSkillStepConfig(
            skill="pdf",
            task="do",
            extra_pip=["pdfplumber>=0.11", "pypdf>=6"],
        )
        assert config.extra_pip == ["pdfplumber>=0.11", "pypdf>=6"]

    def test_llm_agent_execution_mode(self):
        config = AgentSkillStepConfig(
            skill="pdf",
            task="do",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        )
        assert config.execution_mode == AgentSkillExecutionMode.LLM_AGENT
        assert str(config.execution_mode) == "llm_agent"


# ---------------------------------------------------------------------------
# Test 18: SkillResolver — URI parsing and local resolution
# ---------------------------------------------------------------------------

class TestSkillResolver:
    """Tests for the new SkillResolver abstraction."""

    def test_parse_github_uri_with_subpath_and_ref(self):
        from mmar_carl.skill_resolver import _parse_github_uri
        owner, repo, subpath, ref = _parse_github_uri(
            "github://anthropics/skills/skills/pdf@main"
        )
        assert owner == "anthropics"
        assert repo == "skills"
        assert subpath == "skills/pdf"
        assert ref == "main"

    def test_parse_github_uri_default_ref(self):
        from mmar_carl.skill_resolver import _parse_github_uri
        _, _, _, ref = _parse_github_uri("github://anthropics/skills/skills/pptx")
        assert ref == "main"

    def test_parse_github_uri_no_subpath(self):
        from mmar_carl.skill_resolver import _parse_github_uri
        owner, repo, subpath, ref = _parse_github_uri("github://myorg/my-skill@v2")
        assert owner == "myorg"
        assert repo == "my-skill"
        assert subpath == ""
        assert ref == "v2"

    def test_local_resolver_finds_skill(self, tmp_path):
        from mmar_carl.skill_resolver import LocalResolver
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A test skill\n---\nDo stuff.\n"
        )
        resolver = LocalResolver()
        resolved = resolver.resolve(str(skill_dir))
        assert resolved.name == "my-skill"
        assert resolved.frontmatter.description == "A test skill"
        assert resolved.instructions == "Do stuff."
        assert resolved.local_root == skill_dir.resolve()

    def test_local_resolver_uri_syntax(self, tmp_path):
        from mmar_carl.skill_resolver import LocalResolver
        skill_dir = tmp_path / "uri-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: uri-skill\ndescription: URI test\n---\nInstructions.\n"
        )
        resolver = LocalResolver()
        resolved = resolver.resolve(f"local://{skill_dir}")
        assert resolved.name == "uri-skill"

    def test_local_resolver_missing_skill_md(self, tmp_path):
        from mmar_carl.skill_resolver import LocalResolver, SkillResolveError
        resolver = LocalResolver()
        with pytest.raises(SkillResolveError, match="No SKILL.md"):
            resolver.resolve(str(tmp_path / "nonexistent"))

    def test_registry_dispatches_local_path(self, tmp_path):
        from mmar_carl.skill_resolver import SkillResolverRegistry
        skill_dir = tmp_path / "registry-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: registry-skill\ndescription: Registry test\n---\nDo it.\n"
        )
        registry = SkillResolverRegistry()
        resolved = registry.resolve(str(skill_dir))
        assert resolved.name == "registry-skill"

    def test_resolved_skill_sha256_is_set(self, tmp_path):
        """ResolvedSkill should have a non-empty sha256 of SKILL.md."""
        from mmar_carl.skill_resolver import LocalResolver
        import hashlib
        skill_dir = tmp_path / "sha-skill"
        skill_dir.mkdir()
        skill_md_content = "---\nname: sha-skill\ndescription: SHA test\n---\nInstructions.\n"
        (skill_dir / "SKILL.md").write_text(skill_md_content)
        resolver = LocalResolver()
        resolved = resolver.resolve(str(skill_dir))
        expected_sha = hashlib.sha256(skill_md_content.encode()).hexdigest()
        assert resolved.sha256 == expected_sha

    def test_integrity_error_on_wrong_sha256(self, tmp_path):
        """Trust policy sha_pinned should raise SkillIntegrityError on mismatch."""
        from mmar_carl.skill_resolver import LocalResolver, SkillIntegrityError
        skill_dir = tmp_path / "integrity-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: integrity-skill\ndescription: Integrity test\n---\nContent.\n"
        )
        resolver = LocalResolver()
        with pytest.raises(SkillIntegrityError, match="SHA256 mismatch"):
            resolver.resolve(str(skill_dir), sha256="0" * 64)

    def test_resolve_skill_top_level_function(self, tmp_path):
        """Top-level resolve_skill() convenience function works."""
        from mmar_carl.skill_resolver import resolve_skill
        skill_dir = tmp_path / "top-level-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: top-level-skill\ndescription: Top level\n---\nTop level instructions.\n"
        )
        resolved = resolve_skill(str(skill_dir))
        assert resolved.name == "top-level-skill"
        assert resolved.instructions == "Top level instructions."

    def test_resolved_skill_has_scripts(self, tmp_path):
        """ResolvedSkill.scripts should list files in scripts/ subdir."""
        from mmar_carl.skill_resolver import LocalResolver
        skill_dir = tmp_path / "script-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: script-skill\ndescription: Has scripts\n---\nRun scripts.\n"
        )
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        (scripts / "run.py").write_text("print('hi')\n")
        resolver = LocalResolver()
        resolved = resolver.resolve(str(skill_dir))
        script_names = [s.name for s in resolved.scripts]
        assert "run.py" in script_names


# ---------------------------------------------------------------------------
# Test 19: Workspace helpers
# ---------------------------------------------------------------------------

class TestWorkspaceHelpers:
    """Tests for workspace isolation helpers in AgentSkillStepExecutor."""

    def test_create_workspace_makes_in_and_out_dirs(self):
        executor = AgentSkillStepExecutor()
        root, ws_in, ws_out = executor._create_workspace()
        try:
            assert ws_in.is_dir()
            assert ws_out.is_dir()
            assert ws_in.parent == root
            assert ws_out.parent == root
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_stage_input_files_copies_existing_files(self, tmp_path):
        # Create a source file
        src = tmp_path / "input.pdf"
        src.write_bytes(b"PDF content")
        resolved = {"pdf_path": str(src), "mode": "extract"}

        workspace_in = tmp_path / "in"
        workspace_in.mkdir()

        executor = AgentSkillStepExecutor()
        updated = executor._stage_input_files(resolved, workspace_in)

        # File should be copied to workspace_in
        assert updated["pdf_path"] != str(src)
        assert (workspace_in / "input.pdf").exists()
        # Non-file values pass through
        assert updated["mode"] == "extract"

    def test_stage_input_files_passes_through_non_file(self, tmp_path):
        workspace_in = tmp_path / "in"
        workspace_in.mkdir()
        resolved = {"text_input": "some text value", "count": "5"}

        executor = AgentSkillStepExecutor()
        updated = executor._stage_input_files(resolved, workspace_in)

        assert updated == resolved  # No files to stage

    def test_collect_output_files_matches_glob(self, tmp_path):
        ws_out = tmp_path / "out"
        ws_out.mkdir()
        (ws_out / "result.pptx").write_bytes(b"pptx content")
        (ws_out / "data.json").write_bytes(b"{}")
        (ws_out / "notes.txt").write_text("notes")

        executor = AgentSkillStepExecutor()
        # Collect only .pptx files
        files = executor._collect_output_files(ws_out, ["*.pptx"])
        names = [f["name"] for f in files]
        assert "result.pptx" in names
        assert "data.json" not in names

    def test_collect_output_files_all_glob(self, tmp_path):
        ws_out = tmp_path / "out"
        ws_out.mkdir()
        (ws_out / "a.txt").write_text("a")
        (ws_out / "b.json").write_bytes(b"{}")

        executor = AgentSkillStepExecutor()
        files = executor._collect_output_files(ws_out, ["*"])
        assert len(files) == 2

    def test_collect_output_files_empty_dir(self, tmp_path):
        ws_out = tmp_path / "out"
        ws_out.mkdir()
        executor = AgentSkillStepExecutor()
        files = executor._collect_output_files(ws_out, ["*"])
        assert files == []


# ---------------------------------------------------------------------------
# Test 20: LLM_AGENT mode execution
# ---------------------------------------------------------------------------

class TestAgentSkillLLMAgentMode:
    """Tests for the new LLM_AGENT execution mode."""

    @pytest.mark.asyncio
    async def test_llm_agent_final_answer_no_tools(self, tmp_path):
        """When LLM returns no tool calls, loop exits immediately with final answer."""
        skill_dir = make_skill_dir(tmp_path, "agent-skill", "An agent skill", "Process data.")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Analyze the document",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            llm_max_iterations=5,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Agent Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        # get_response_with_tools returns (content, []) — final answer immediately
        mock_llm.get_response_with_tools = AsyncMock(
            return_value=("Final analysis result", [])
        )
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert "Final analysis result" in result.result
        assert result.result_data.get("execution_mode") == "llm_agent"
        assert result.result_data.get("iterations") == 1

    @pytest.mark.asyncio
    async def test_llm_agent_tool_call_then_final(self, tmp_path, tmp_path_factory):
        """LLM calls one tool then returns final answer."""
        skill_dir = tmp_path / "tool-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: tool-skill\ndescription: Uses tools\n---\nUse run_script.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "echo.py").write_text("print('script output')\n")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Process something",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            python_executable=sys.executable,
            llm_max_iterations=10,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Tool Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        # First call: returns a tool call; second call: returns final answer
        mock_llm.get_response_with_tools = AsyncMock(side_effect=[
            (
                "",
                [{"id": "call_1", "name": "run_script", "arguments": {
                    "script_path": "scripts/echo.py",
                    "args": [],
                }}],
            ),
            ("Final result after using tool", []),
        ])
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        assert "Final result after using tool" in result.result
        assert result.result_data.get("tool_calls_made") == 1
        assert result.result_data.get("iterations") == 2

    @pytest.mark.asyncio
    async def test_llm_agent_list_resources_tool(self, tmp_path):
        """list_resources tool should return skill script and reference names."""
        skill_dir = tmp_path / "listed-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: listed-skill\ndescription: Listed\n---\nContent.\n"
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "main.py").write_text("print('hi')\n")
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "GUIDE.md").write_text("Guide content\n")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="List resources",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        )

        step = MagicMock()
        step.number = 1
        step.title = "List Step"
        step.step_config = config

        tool_results: list = []

        context = make_mock_context()
        mock_llm = AsyncMock()

        async def capture_tool_result(*args, **kwargs):
            messages = kwargs.get("messages") or []
            # First turn: call list_resources
            if len(messages) <= 2:
                return ("", [{"id": "c1", "name": "list_resources", "arguments": {}}])
            # After tool result: return final
            for msg in messages:
                if msg.get("role") == "tool":
                    tool_results.append(msg.get("content", ""))
            return ("Resources listed.", [])

        mock_llm.get_response_with_tools = AsyncMock(side_effect=capture_tool_result)
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        # Tool result should contain the script name
        assert any("main.py" in r for r in tool_results)

    @pytest.mark.asyncio
    async def test_llm_agent_write_file_creates_output(self, tmp_path):
        """write_file tool should create a file in workspace/out."""
        skill_dir = make_skill_dir(tmp_path, "writer-skill", "Writer skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Write something",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            output_capture="both",
        )

        step = MagicMock()
        step.number = 1
        step.title = "Writer Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_tools = AsyncMock(side_effect=[
            (
                "",
                [{"id": "w1", "name": "write_file", "arguments": {
                    "filename": "result.txt",
                    "content": "Hello from LLM_AGENT",
                }}],
            ),
            ("Done.", []),
        ])
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        # output_files should contain result.txt
        output_files = result.result_data.get("output_files", [])
        names = [f["name"] for f in output_files]
        assert "result.txt" in names

    @pytest.mark.asyncio
    async def test_llm_agent_max_iterations(self, tmp_path):
        """When max_iterations is reached, should still return a result."""
        skill_dir = make_skill_dir(tmp_path, "loop-skill", "Loop skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Loop forever",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            llm_max_iterations=3,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Loop Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        # Always returns a tool call — never a final answer
        mock_llm.get_response_with_tools = AsyncMock(
            return_value=(
                "thinking...",
                [{"id": "loop", "name": "list_resources", "arguments": {}}],
            )
        )
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        # Should complete (not raise) and respect max iterations
        assert result.success is True
        assert result.result_data.get("iterations") == 3

    @pytest.mark.asyncio
    async def test_llm_agent_history_entry(self, tmp_path):
        """LLM_AGENT steps should create a history entry with [skill:name] tag."""
        skill_dir = make_skill_dir(tmp_path, "history-skill", "History skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Do something",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        )

        step = MagicMock()
        step.number = 1
        step.title = "History Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_tools = AsyncMock(return_value=("Result text", []))
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success is True
        history_entry = result.updated_history[-1]
        assert "[skill:history-skill]" in history_entry


# ---------------------------------------------------------------------------
# Test: extra_pip — package installation overlay
# ---------------------------------------------------------------------------

class TestExtraPipOverlay:
    """Tests for the extra_pip → PYTHONPATH overlay mechanism."""

    def test_build_subprocess_env_without_overlay(self):
        """When pip_overlay is None, _build_subprocess_env returns None."""
        env = AgentSkillStepExecutor._build_subprocess_env(None)
        assert env is None

    def test_build_subprocess_env_prepends_pythonpath(self, tmp_path):
        """pip_overlay is prepended to PYTHONPATH in the returned env dict."""
        overlay = tmp_path / "pip_overlay"
        overlay.mkdir()
        env = AgentSkillStepExecutor._build_subprocess_env(overlay)
        assert env is not None
        assert env["PYTHONPATH"].startswith(str(overlay))
        # Must also contain everything from the current process env (e.g. PATH)
        assert "PATH" in env

    def test_build_subprocess_env_preserves_existing_pythonpath(self, tmp_path, monkeypatch):
        """Existing PYTHONPATH entries are preserved after the overlay prefix."""
        monkeypatch.setenv("PYTHONPATH", "/existing/path")
        overlay = tmp_path / "overlay"
        overlay.mkdir()
        env = AgentSkillStepExecutor._build_subprocess_env(overlay)
        assert env is not None
        assert "/existing/path" in env["PYTHONPATH"]
        assert env["PYTHONPATH"].startswith(str(overlay))

    @pytest.mark.asyncio
    async def test_execute_script_mode_uses_overlay_pythonpath(self, tmp_path):
        """
        A script that imports a module from the overlay dir succeeds.
        We create a fake package directly in the overlay dir (no pip needed).
        """
        # Create a minimal skill
        skill_dir = tmp_path / "myscill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: myscill\ndescription: Test skill\n---\nInstructions.\n"
        )
        script_path = skill_dir / "run.py"
        script_path.write_text(
            "import my_fake_pkg\nprint(my_fake_pkg.value)\n"
        )

        # Create the overlay dir with a fake package
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()
        (overlay_dir / "my_fake_pkg.py").write_text(
            "value = 'injected-by-overlay'\n"
        )

        from mmar_carl.skill_loader import SkillLoader
        from mmar_carl.models.agent_skill import AgentSkillSource, AgentSkillStepConfig, AgentSkillExecutionMode

        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=False)
        manifest = loader.load_sync(source)

        config = AgentSkillStepConfig(
            skill=source,
            task="test",
            execution_mode=AgentSkillExecutionMode.SCRIPT,
            script_name="run.py",
        )

        executor = AgentSkillStepExecutor()
        stdout, stderr, rc = await executor._execute_script_mode(
            config, manifest, resolved_inputs={}, pip_overlay=overlay_dir
        )

        assert rc == 0, f"Script failed (rc={rc}):\nstdout={stdout}\nstderr={stderr}"
        assert "injected-by-overlay" in stdout

    @pytest.mark.asyncio
    async def test_execute_script_mode_without_overlay_cannot_import(self, tmp_path):
        """
        Without pip_overlay, a script importing a non-installed package fails.
        """
        skill_dir = tmp_path / "badimport"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: badimport\ndescription: Test\n---\nInstructions.\n"
        )
        script_path = skill_dir / "run.py"
        script_path.write_text(
            "import definitely_not_installed_xyzzy123\nprint('ok')\n"
        )

        from mmar_carl.skill_loader import SkillLoader
        from mmar_carl.models.agent_skill import AgentSkillSource, AgentSkillStepConfig, AgentSkillExecutionMode

        source = AgentSkillSource(path=str(skill_dir))
        loader = SkillLoader(enable_cache=False)
        manifest = loader.load_sync(source)

        config = AgentSkillStepConfig(
            skill=source, task="test",
            execution_mode=AgentSkillExecutionMode.SCRIPT,
            script_name="run.py",
        )

        executor = AgentSkillStepExecutor()
        stdout, stderr, rc = await executor._execute_script_mode(
            config, manifest, resolved_inputs={}, pip_overlay=None
        )

        assert rc != 0, "Expected non-zero exit code for missing import"

    @pytest.mark.asyncio
    async def test_full_execute_with_extra_pip_installs_before_script(self, tmp_path):
        """
        Full execute() path: extra_pip is non-empty →
        _install_extra_pip is called, then script runs with overlay PYTHONPATH.
        We mock _install_extra_pip to create the fake package directly,
        simulating what pip would do.
        """
        from unittest.mock import patch
        from mmar_carl.models.agent_skill import AgentSkillSource, AgentSkillStepConfig, AgentSkillExecutionMode
        from mmar_carl.models.steps import AgentSkillStepDescription

        skill_dir = tmp_path / "pipskill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: pipskill\ndescription: Test pip skill\n---\nInstructions.\n"
        )
        (skill_dir / "run.py").write_text(
            "import pip_injected\nprint(pip_injected.msg)\n"
        )

        context = make_mock_context()
        context.get_llm_client_for_step = MagicMock()

        step = AgentSkillStepDescription(
            number=1,
            title="Pip test",
            config=AgentSkillStepConfig(
                skill=AgentSkillSource(path=str(skill_dir)),
                task="test",
                execution_mode=AgentSkillExecutionMode.SCRIPT,
                script_name="run.py",
                extra_pip=["pip_injected"],
            ),
        )

        # patch.object replaces the @staticmethod descriptor with a plain function,
        # so the instance's self is prepended when called via self._install_extra_pip().
        async def fake_install_extra_pip(_ignored_self, packages, target_dir):
            # Write the fake module directly, simulating what pip install would do
            (target_dir / "pip_injected.py").write_text(
                "msg = 'installed-via-extra-pip'\n"
            )

        executor = AgentSkillStepExecutor()

        with patch.object(AgentSkillStepExecutor, '_install_extra_pip', new=fake_install_extra_pip):
            result = await executor.execute(step, context)

        assert result.success, f"Execution failed: {result.error_message}"
        assert "installed-via-extra-pip" in result.result


# ---------------------------------------------------------------------------
# Test 19: GitHub URI → GithubResolver routing (no git-clone)
# ---------------------------------------------------------------------------


class TestGithubURIResolution:
    """
    _resolve_git() must route github.com git_url values through GithubResolver
    (tarball download, no git binary) instead of subprocess git clone.
    """

    def _make_skill_dir(self, directory: Path) -> Path:
        """Create a minimal SKILL.md in directory (created if needed) and return the path."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: Test.\n---\n# Instructions\nDo stuff.\n"
        )
        return directory

    def test_github_url_routes_to_github_resolver_not_git_clone(self, tmp_path):
        """
        When git_url starts with https://github.com/, _resolve_git() must call
        GithubResolver.resolve() and NOT subprocess git clone.
        """
        from unittest.mock import patch, MagicMock
        from mmar_carl.skill_resolver import ResolvedSkill
        from mmar_carl.skill_loader import SkillLoader
        from mmar_carl.models.agent_skill import AgentSkillSource

        skill_dir = self._make_skill_dir(tmp_path / "skill")
        mock_resolved = MagicMock(spec=ResolvedSkill)
        mock_resolved.local_root = skill_dir

        source = AgentSkillSource(
            git_url="https://github.com/anthropics/skills",
            git_ref="main",
            git_subdirectory="skills/pdf",
        )

        loader = SkillLoader()
        with patch("mmar_carl.skill_loader.SkillLoader._resolve_git_via_github_resolver",
                   return_value=str(skill_dir)) as mock_gh, \
             patch("subprocess.run") as mock_git:
            result = loader._resolve_git(source)

        mock_gh.assert_called_once_with(source)
        mock_git.assert_not_called()  # git clone must NOT be called
        assert result == str(skill_dir)

    def test_non_github_url_still_uses_git_clone(self, tmp_path):
        """
        When git_url points to a non-GitHub host, fallback to git clone as before.
        Uses a fresh tmp_path cache dir to avoid hitting a stale cached clone.
        """
        from unittest.mock import patch
        from mmar_carl.skill_loader import SkillLoader
        from mmar_carl.models.agent_skill import AgentSkillSource

        source = AgentSkillSource(
            git_url="https://gitlab.example.com/org/skills",
            git_ref="main",
            git_subdirectory="pdf",
        )

        def fake_run(cmd, **kwargs):
            # Simulate git clone: write SKILL.md into the target clone dir
            clone_dir = cmd[-1]  # last arg to git clone
            import os
            os.makedirs(os.path.join(clone_dir, "pdf"), exist_ok=True)
            with open(os.path.join(clone_dir, "pdf", "SKILL.md"), "w") as f:
                f.write("---\nname: t\ndescription: d\n---\n# B\n")
            return MagicMock(returncode=0)

        # Use tmp_path as cache so there's no pre-existing cached clone
        loader = SkillLoader(cache_dir=str(tmp_path / "cache"))
        with patch("subprocess.run", side_effect=fake_run) as mock_git, \
             patch("mmar_carl.skill_loader.SkillLoader._resolve_git_via_github_resolver") as mock_gh:
            result = loader._resolve_git(source)

        mock_gh.assert_not_called()  # GithubResolver NOT used for non-GitHub
        mock_git.assert_called_once()
        assert result is not None

    def test_github_uri_uri_reconstruction(self, tmp_path):
        """
        _resolve_git_via_github_resolver() correctly reconstructs the github:// URI
        from AgentSkillSource fields and calls GithubResolver.resolve().
        """
        from unittest.mock import patch, MagicMock
        from mmar_carl.skill_resolver import ResolvedSkill
        from mmar_carl.skill_loader import SkillLoader
        from mmar_carl.models.agent_skill import AgentSkillSource

        skill_dir = self._make_skill_dir(tmp_path / "skill")
        mock_resolved = MagicMock(spec=ResolvedSkill)
        mock_resolved.local_root = skill_dir

        source = AgentSkillSource(
            git_url="https://github.com/anthropics/skills",
            git_ref="main",
            git_subdirectory="skills/pdf",
        )

        loader = SkillLoader()
        with patch("mmar_carl.skill_resolver.GithubResolver") as MockGH:
            MockGH.return_value.resolve.return_value = mock_resolved
            result = loader._resolve_git_via_github_resolver(source)

        # The URI passed to GithubResolver.resolve() must be correctly formed
        called_uri = MockGH.return_value.resolve.call_args[0][0]
        assert called_uri == "github://anthropics/skills/skills/pdf@main"
        assert result == str(skill_dir)

    def test_github_uri_no_subpath_reconstruction(self, tmp_path):
        """URI with no subpath: github://owner/repo@ref (no trailing slash)."""
        from unittest.mock import patch, MagicMock
        from mmar_carl.skill_resolver import ResolvedSkill
        from mmar_carl.skill_loader import SkillLoader
        from mmar_carl.models.agent_skill import AgentSkillSource

        skill_dir = self._make_skill_dir(tmp_path / "skill")
        mock_resolved = MagicMock(spec=ResolvedSkill)
        mock_resolved.local_root = skill_dir

        source = AgentSkillSource(
            git_url="https://github.com/myorg/my-skill",
            git_ref="HEAD",
            git_subdirectory=None,
        )

        loader = SkillLoader()
        with patch("mmar_carl.skill_resolver.GithubResolver") as MockGH:
            MockGH.return_value.resolve.return_value = mock_resolved
            result = loader._resolve_git_via_github_resolver(source)

        called_uri = MockGH.return_value.resolve.call_args[0][0]
        assert called_uri == "github://myorg/my-skill@HEAD"
        assert result == str(skill_dir)

    def test_github_resolver_error_wrapped_as_skill_not_found(self, tmp_path):
        """SkillResolveError from GithubResolver is re-raised as SkillNotFoundError."""
        from unittest.mock import patch
        from mmar_carl.skill_resolver import SkillResolveError
        from mmar_carl.skill_loader import SkillLoader, SkillNotFoundError
        from mmar_carl.models.agent_skill import AgentSkillSource

        source = AgentSkillSource(
            git_url="https://github.com/bad/repo",
            git_ref="main",
            git_subdirectory=None,
        )

        loader = SkillLoader()
        with patch("mmar_carl.skill_resolver.GithubResolver") as MockGH:
            MockGH.return_value.resolve.side_effect = SkillResolveError("network error")
            with pytest.raises(SkillNotFoundError, match="GithubResolver"):
                loader._resolve_git_via_github_resolver(source)


# ---------------------------------------------------------------------------
# Test 21: LLM_AGENT parallel tool call execution
# ---------------------------------------------------------------------------


class TestLLMAgentParallelToolCalls:
    """
    Verify that asyncio.gather in _execute_llm_agent_mode correctly executes
    multiple tool calls concurrently and preserves their order in history.
    """

    @pytest.mark.asyncio
    async def test_parallel_write_file_calls_both_succeed(self, tmp_path):
        """
        Two concurrent write_file calls to different filenames must both
        succeed and produce independent files in workspace/out.
        """
        skill_dir = make_skill_dir(tmp_path, "parallel-skill", "Parallel tool skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Write two files concurrently",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            output_capture="files",
        )

        step = MagicMock()
        step.number = 1
        step.title = "Parallel writes"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()

        # First LLM turn: return two concurrent write_file tool calls
        # Second LLM turn: return final answer
        mock_llm.get_response_with_tools = AsyncMock(side_effect=[
            (
                "",
                [
                    {"id": "w1", "name": "write_file", "arguments": {
                        "filename": "file_a.txt",
                        "content": "Content A",
                    }},
                    {"id": "w2", "name": "write_file", "arguments": {
                        "filename": "file_b.txt",
                        "content": "Content B",
                    }},
                ],
            ),
            ("Both files written.", []),
        ])
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success, f"Execution failed: {result.error_message}"

        output_files: list[dict] = result.result_data.get("output_files", [])
        output_names = {f["name"] for f in output_files}
        assert "file_a.txt" in output_names, f"file_a.txt missing; got {output_names}"
        assert "file_b.txt" in output_names, f"file_b.txt missing; got {output_names}"
        # Both files appear as separate entries (parallel writes didn't overwrite each other)
        assert len(output_files) == 2

    @pytest.mark.asyncio
    async def test_parallel_tool_results_order_matches_tool_calls(self, tmp_path):
        """
        After asyncio.gather, tool result messages must be appended to history
        in the same order as the original tool calls (not gather completion order).
        """
        skill_dir = make_skill_dir(tmp_path, "order-skill", "Order-check skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Check order",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Order check"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()

        # Capture the history messages passed to the second LLM call so we can
        # inspect the tool result ordering.
        captured_histories: list = []

        async def capture_and_respond(system_prompt, user_prompt, tools, messages):
            captured_histories.append(list(messages))
            if len(captured_histories) == 1:
                # First call: return two tool calls
                return (
                    "",
                    [
                        {"id": "tc_alpha", "name": "list_resources", "arguments": {}},
                        {"id": "tc_beta",  "name": "list_resources", "arguments": {}},
                    ],
                )
            # Second call: final answer
            return ("Ordering verified.", [])

        mock_llm.get_response_with_tools = AsyncMock(side_effect=capture_and_respond)
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success, f"Execution failed: {result.error_message}"
        assert len(captured_histories) >= 2, "Expected at least 2 LLM calls"

        # The second LLM call receives history including tool results.
        # Tool result messages should appear in the same order as the tool calls
        # (tc_alpha before tc_beta).
        second_history = captured_histories[1]
        tool_result_msgs = [m for m in second_history if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 2, f"Expected 2 tool result messages; got {tool_result_msgs}"
        assert tool_result_msgs[0]["tool_call_id"] == "tc_alpha"
        assert tool_result_msgs[1]["tool_call_id"] == "tc_beta"

    @pytest.mark.asyncio
    async def test_single_tool_call_still_works(self, tmp_path):
        """gather with a single tool call should behave identically to the old sequential path."""
        skill_dir = make_skill_dir(tmp_path, "single-skill", "Single tool skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="List resources once",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Single tool"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_tools = AsyncMock(side_effect=[
            (
                "",
                [{"id": "single", "name": "list_resources", "arguments": {}}],
            ),
            ("Done.", []),
        ])
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success
        assert result.result_data.get("tool_calls_made") == 1

    @pytest.mark.asyncio
    async def test_parallel_calls_count_tracked_correctly(self, tmp_path):
        """tool_calls_made metadata should count ALL tool calls across both gather batches."""
        skill_dir = make_skill_dir(tmp_path, "count-skill", "Count tool calls")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Make multiple tool calls",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            llm_max_iterations=5,
        )

        step = MagicMock()
        step.number = 1
        step.title = "Count check"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()

        # Iteration 1: 3 parallel tool calls
        # Iteration 2: 2 parallel tool calls
        # Iteration 3: final answer
        mock_llm.get_response_with_tools = AsyncMock(side_effect=[
            (
                "",
                [
                    {"id": "a1", "name": "list_resources", "arguments": {}},
                    {"id": "a2", "name": "list_resources", "arguments": {}},
                    {"id": "a3", "name": "list_resources", "arguments": {}},
                ],
            ),
            (
                "",
                [
                    {"id": "b1", "name": "list_resources", "arguments": {}},
                    {"id": "b2", "name": "list_resources", "arguments": {}},
                ],
            ),
            ("All done.", []),
        ])
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success
        assert result.result_data.get("tool_calls_made") == 5
        assert result.result_data.get("iterations") == 3


# ---------------------------------------------------------------------------
# Test: persist_workspace
# ---------------------------------------------------------------------------

class TestPersistWorkspace:
    """
    Verify persist_workspace=True skips workspace cleanup and writes path to memory.
    """

    @pytest.mark.asyncio
    async def test_no_workspace_memory_key_when_persist_false(self, tmp_path):
        """When persist_workspace=False (default), no workspace key is written to memory."""
        skill_dir = make_skill_dir(tmp_path, "cleanup-skill", "Cleanup test skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Do something",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            persist_workspace=False,
            output_memory_key="my_key",
        )

        step = MagicMock()
        step.number = 1
        step.title = "Cleanup Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_tools = AsyncMock(return_value=("done", []))
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success
        # No workspace key in memory when persist_workspace=False
        # (mock context stores memory in metadata as _mem_{namespace}_{key})
        assert "_mem_agent_skill_my_key_workspace" not in context.metadata
        # Also no persisted_workspace in result_data
        assert result.result_data.get("persisted_workspace") is None

    @pytest.mark.asyncio
    async def test_workspace_preserved_when_persist_workspace_true(self, tmp_path):
        """When persist_workspace=True, workspace dir survives after execution."""
        skill_dir = make_skill_dir(tmp_path, "persist-skill", "Persist test skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Do something persistent",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            persist_workspace=True,
            output_memory_key="my_result",
        )

        step = MagicMock()
        step.number = 1
        step.title = "Persist Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_tools = AsyncMock(return_value=("persistent result", []))
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success, result.error_message

        ws_path_str = result.result_data.get("persisted_workspace") or result.result_data.get("workspace_root")
        assert ws_path_str is not None, "persisted_workspace should be in result_data"

        ws_path = Path(ws_path_str)
        try:
            assert ws_path.exists(), f"Workspace dir should still exist: {ws_path}"
        finally:
            import shutil
            shutil.rmtree(ws_path, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_workspace_path_written_to_memory(self, tmp_path):
        """Persisted workspace path is written to memory.agent_skill.<key>_workspace."""
        skill_dir = make_skill_dir(tmp_path, "mem-skill", "Memory test skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Write result",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            persist_workspace=True,
            output_memory_key="analysis",
        )

        step = MagicMock()
        step.number = 1
        step.title = "Mem Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_tools = AsyncMock(return_value=("analysis output", []))
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success, result.error_message

        # mock context stores memory in metadata as _mem_{namespace}_{key}
        mem_key = "_mem_agent_skill_analysis_workspace"
        assert mem_key in context.metadata, (
            f"Expected '{mem_key}' in context.metadata; got keys: {[k for k in context.metadata if k.startswith('_mem')]}"
        )

        ws_path = Path(context.metadata[mem_key])
        try:
            assert ws_path.exists(), f"Workspace should still exist at: {ws_path}"
        finally:
            import shutil
            shutil.rmtree(ws_path, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_workspace_path_uses_title_when_no_output_memory_key(self, tmp_path):
        """When output_memory_key is not set, workspace memory key uses step title."""
        skill_dir = make_skill_dir(tmp_path, "title-skill", "Title key skill")

        config = AgentSkillStepConfig(
            skill=AgentSkillSource(path=str(skill_dir)),
            task="Write result",
            execution_mode=AgentSkillExecutionMode.LLM_AGENT,
            persist_workspace=True,
        )

        step = MagicMock()
        step.number = 1
        step.title = "My Title Step"
        step.step_config = config

        context = make_mock_context()
        mock_llm = AsyncMock()
        mock_llm.get_response_with_tools = AsyncMock(return_value=("result", []))
        context.get_llm_client_for_step = MagicMock(return_value=mock_llm)

        executor = AgentSkillStepExecutor()
        result = await executor.execute(step, context)

        assert result.success, result.error_message

        # mock context stores memory in metadata as _mem_{namespace}_{key}
        mem_key = "_mem_agent_skill_My Title Step_workspace"
        assert mem_key in context.metadata, (
            f"Expected '{mem_key}' in context.metadata; got: {[k for k in context.metadata if k.startswith('_mem')]}"
        )

        ws_path = Path(context.metadata[mem_key])
        try:
            assert ws_path.exists()
        finally:
            import shutil
            shutil.rmtree(ws_path, ignore_errors=True)

    def test_persist_workspace_field_defaults_false(self):
        """persist_workspace field defaults to False."""
        config = AgentSkillStepConfig(skill="my-skill", task="do something")
        assert config.persist_workspace is False

    def test_persist_workspace_can_be_set_true(self):
        """persist_workspace field can be set to True."""
        config = AgentSkillStepConfig(
            skill="my-skill",
            task="do something",
            persist_workspace=True,
        )
        assert config.persist_workspace is True
