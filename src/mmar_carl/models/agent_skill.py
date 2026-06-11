"""
AgentSkill models for CARL reasoning system.

Defines configuration models for the AgentSkillStep type, which integrates
the AgentSkills open format (https://agentskills.io) into CARL reasoning chains.
"""

from enum import StrEnum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Imported here to avoid circular imports (config.py does not import agent_skill.py)
from .config import LLMStepConfig  # noqa: E402


class AgentSkillExecutionMode(StrEnum):
    """How to execute an AgentSkill within a CARL chain."""

    LLM = "llm"
    """Inject SKILL.md instructions as system prompt, call LLM once for the task (single-shot)."""

    SCRIPT = "script"
    """Run a bundled script directly without any LLM call (deterministic)."""

    HYBRID = "hybrid"
    """Try script first; fall back to LLM if script fails or returns empty output."""

    SUBAGENT = "subagent"
    """Run script to collect raw data, then LLM synthesises the result (data-driven)."""

    LLM_AGENT = "llm_agent"
    """
    Iterative tool-calling loop matching the AgentSkills spec's progressive-disclosure model.

    The LLM receives SKILL.md as system prompt plus a constrained tool surface:
      - ``run_script(script_path, args)``  — execute a bundled script
      - ``read_file(path)``                — read a file from workspace or skill dir
      - ``write_file(path, content)``      — write a file to /workspace/out
      - ``list_resources()``               — list available scripts and references
      - ``read_resource(resource_path)``   — read a reference/asset file

    The loop continues until the LLM stops calling tools (final answer) or
    ``llm_max_iterations`` is reached.  Input files are staged in ``/workspace/in``
    and outputs are collected from ``/workspace/out``.

    This mode requires the LLM client to support OpenAI-style function calling.
    Falls back to single-shot LLM mode for clients that do not.
    """


class AgentSkillSource(BaseModel):
    """Describes where to find an AgentSkill.

    Supports four resolution strategies (exactly one must be set):
    - ``path``: local filesystem path to the skill directory
    - ``name``: skill name, searched in standard directories
    - ``git_url``: Git repository URL (cloned via ``git clone``)
    - ``package``: installed Python package

    For URI-based resolution (``github://``, ``https://``, ``module://``),
    use ``AgentSkillStepConfig.skill`` as a URI string — it will be auto-parsed
    into the appropriate ``AgentSkillSource`` fields and resolved via
    :mod:`mmar_carl.skill_resolver`.
    """
    """Describes where to find an AgentSkill."""

    # Option A: explicit local path to the skill directory (highest priority)
    path: Optional[str] = None

    # Option B: skill name — searched in search_paths + standard locations
    name: Optional[str] = None

    # Option C: Git repository URL (cloned into ~/.cache/mmar_carl/skills/<repo>/)
    # Format: "https://github.com/org/repo" with optional git_subdirectory for subdirectory
    git_url: Optional[str] = None
    git_subdirectory: Optional[str] = None  # e.g. "skills/pdf"
    git_ref: str = "HEAD"  # branch/tag/commit

    # Option D: installed Python package name
    package: Optional[str] = None
    package_subpath: Optional[str] = None  # path within the package

    # Additional search paths (prepended before standard ~/.agents/skills/ etc.)
    search_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self) -> "AgentSkillSource":
        sources = [self.path, self.name, self.git_url, self.package]
        if sum(s is not None for s in sources) != 1:
            raise ValueError("Exactly one of path/name/git_url/package must be set")
        return self


class SkillManifest(BaseModel):
    """Parsed content of a SKILL.md file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Frontmatter fields
    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[str] = None   # raw string from frontmatter, e.g. "Bash(git:*) Read Write"
    metadata: dict[str, str] = Field(default_factory=dict)

    # Parsed body (markdown after frontmatter)
    instructions: str

    # Filesystem info
    skill_dir: str    # absolute path to the skill root directory
    skill_md_path: str  # absolute path to SKILL.md

    # Discovered bundled resources
    scripts: list[str] = Field(default_factory=list)     # relative paths like "scripts/extract.py"
    references: list[str] = Field(default_factory=list)  # relative paths like "references/REFERENCE.md"
    assets: list[str] = Field(default_factory=list)      # relative paths like "assets/template.pptx"

    def get_allowed_tools(self) -> list[str]:
        """
        Parse the `allowed-tools` frontmatter field into a list of tool tokens.

        The AgentSkills spec uses space-separated tokens, each optionally followed
        by a parenthesised constraint, e.g.:
            "Bash(git:*) Read Write Edit"
        returns ["Bash(git:*)", "Read", "Write", "Edit"]

        Returns an empty list when no `allowed-tools` field is present, meaning
        the skill has no tool restrictions.
        """
        if not self.allowed_tools:
            return []
        return self.allowed_tools.split()

    def get_allowed_tool_names(self) -> list[str]:
        """
        Return bare tool names (without constraints) from `allowed-tools`.

        "Bash(git:*) Read Write" -> ["Bash", "Read", "Write"]
        """
        import re
        return [re.sub(r"\(.*\)$", "", tok) for tok in self.get_allowed_tools()]

    def restricts_tools(self) -> bool:
        """Return True if this skill declares tool restrictions."""
        return bool(self.allowed_tools)

    def to_memory_payload(
        self,
        *,
        uri: Optional[str] = None,
        sha256: Optional[str] = None,
        tarball_url: Optional[str] = None,
        tarball_sha256: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compact dict suitable for a gigaevo-memory ``agent_skill`` card.

        The returned dict has the shape Memory expects:

        * ``name``, ``description`` (sanitised), ``uri``, ``sha256``
        * ``manifest`` — full frontmatter ``metadata`` dict
        * ``instructions`` — markdown body below frontmatter
        * ``allowed_tools`` — parsed list from ``get_allowed_tools()``
        * ``tags`` — pulled from ``metadata["tags"]`` if present
        * ``compatibility`` — semver / range string from frontmatter
        * ``tarball_url`` / ``tarball_sha256`` — github:// origin info
          (None for skills resolved from local paths or modules)

        Args:
            uri: Origin URI of this skill (e.g.
                ``"github://anthropics/skills/skills/pdf@main"``).
                Optional — None when the skill was loaded from a plain
                local path.
            sha256: SHA-256 of the SKILL.md content. ``ResolvedSkill``
                already tracks this; ``SkillLoader.catalog_all`` may
                pass it through when known.
            tarball_url / tarball_sha256: github:// resolver byproducts.

        ``tags`` is parsed permissively from ``metadata["tags"]``:
        accepts a list, a comma-separated string, or a single string
        (wrapped to a one-element list). Missing / empty → ``[]``.
        """
        raw_tags = self.metadata.get("tags") if self.metadata else None
        tags: list[str]
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags if t is not None]
        elif isinstance(raw_tags, str) and raw_tags.strip():
            # Comma-separated string is the common YAML frontmatter shape.
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        else:
            tags = []

        return {
            "name": self.name,
            "description": _sanitize_description(self.description),
            "uri": uri,
            "sha256": sha256,
            "manifest": dict(self.metadata) if self.metadata else {},
            "instructions": self.instructions,
            "allowed_tools": self.get_allowed_tools(),
            "tags": tags,
            "compatibility": self.compatibility,
            "tarball_url": tarball_url,
            "tarball_sha256": tarball_sha256,
        }


def _sanitize_description(text: Optional[str]) -> str:
    """Light scrub for a description string before persistence.

    * Coerces ``None`` to empty string.
    * Strips surrounding whitespace.
    * Removes ASCII control characters (newlines stay — they're fine
      in card descriptions).
    * Caps length at 500 chars with a ``…`` ellipsis when overflowing
      so the card stays diff-friendly.
    """
    if not text:
        return ""
    s = str(text).strip()
    # Remove control chars except newline (0x0a) and tab (0x09).
    s = "".join(ch for ch in s if ord(ch) >= 0x20 or ch in "\n\t")
    if len(s) > 500:
        s = s[:499] + "…"
    return s


class AgentSkillStepConfig(BaseModel):
    """Configuration for AgentSkillStep.

    Skill identity
    --------------
    ``skill`` accepts any of these forms:

    * ``"pdf"``                                     — name lookup in standard dirs
    * ``"/abs/path/to/skill"``                      — local filesystem path
    * ``"github://anthropics/skills/skills/pdf"``   — GitHub tarball (recommended)
    * ``"github://anthropics/skills/skills/pdf@main"``
    * ``"https://example.com/skill.tar.gz"``        — generic tarball
    * ``"module://my_pkg.skills.pdf"``              — Python package
    * ``AgentSkillSource(git_url="...", ...)``       — explicit source object

    Trust & integrity
    -----------------
    ``trust_policy="sha_pinned"`` requires ``skill_sha256`` and verifies that
    the SKILL.md file matches.  Default ``"any"`` skips verification.
    """

    # Which skill to use
    skill: Union[AgentSkillSource, str] = Field(
        ...,
        description=(
            "Skill to use. Plain strings are auto-coerced: "
            "'pdf' → name lookup, '/path' → path, "
            "'github://owner/repo/path[@ref]' → GitHub tarball, "
            "'https://...' → HTTPS tarball, 'module://pkg' → Python package."
        ),
    )

    # What to do with the skill
    task: str = Field(
        ...,
        description="The specific task to perform using the skill. Becomes the user turn in LLM mode.",
    )

    # Execution strategy
    execution_mode: AgentSkillExecutionMode = Field(
        default=AgentSkillExecutionMode.LLM,
        description="How to execute the skill",
    )

    # Input wiring (maps skill-meaningful names -> CARL context references)
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps descriptive input names to CARL context refs. "
            "Keys are interpolated into the task string via {key} placeholders. "
            "Values use CARL reference syntax ($history[-1], $memory.ns.key, etc.)"
        ),
    )

    # Where to store the step result
    output_key: str = Field(default="result", description="Key label for result in history entry")
    output_memory_key: Optional[str] = Field(
        default=None,
        description=(
            "If set, write the skill's text result to this memory key under the "
            "'agent_skill' namespace. Useful for wiring parallel skill branches: "
            "each branch writes to a different key, and a downstream step reads them "
            "all via $memory.agent_skill.<key>."
        ),
    )
    output_memory_namespace: str = Field(
        default="agent_skill",
        description="Memory namespace for output_memory_key writes (default: 'agent_skill').",
    )
    output_file_key: Optional[str] = Field(
        default=None,
        description=(
            "If set, write the path of any output file produced by the skill to "
            "memory key 'agent_skill.<output_file_key>'"
        ),
    )

    # LLM settings (for LLM and HYBRID modes)
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM config override (model, temperature, max_tokens). Uses context default if None.",
    )
    system_prompt_prefix: Optional[str] = Field(
        default=None,
        description="Text prepended before skill instructions in the system prompt",
    )
    include_skill_description: bool = Field(
        default=True,
        description="Whether to include the skill description (from frontmatter) in the LLM prompt",
    )
    max_instructions_chars: Optional[int] = Field(
        default=3000,
        description=(
            "Maximum characters of skill instructions to include in the LLM system prompt. "
            "Skill SKILL.md files can be very large (8-9KB+); truncating prevents ToS rejections "
            "from providers that flag large code-heavy prompts. Set to None for no limit."
        ),
    )
    strip_code_blocks: bool = Field(
        default=True,
        description=(
            "Strip fenced code blocks (```...```) from skill instructions before sending to LLM. "
            "Code examples in SKILL.md are guides for the LLM's reasoning, not needed verbatim, "
            "and including them can trigger provider ToS filters."
        ),
    )
    filter_security_terms: bool = Field(
        default=True,
        description=(
            "Remove markdown sections whose headings mention security-sensitive terms "
            "(password, encrypt, decrypt) from skill instructions and description. "
            "Prevents provider ToS filter rejections caused by security-related content "
            "in skill documentation (e.g. PDF decryption instructions)."
        ),
    )

    # Script settings (for SCRIPT and HYBRID modes)
    script_name: Optional[str] = Field(
        default=None,
        description="Relative path to script within skill dir (e.g. 'scripts/extract.py'). Auto-detected if None.",
    )
    script_args: dict[str, str] = Field(
        default_factory=dict,
        description="Static CLI args for script execution. Merged with resolved input_mapping values.",
    )
    python_executable: str = Field(
        default="python",
        description="Python interpreter to use for running .py scripts",
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="Working directory for script execution. Defaults to skill_dir.",
    )

    # Cache settings
    cache_skills: bool = Field(
        default=True,
        description="Cache loaded SkillManifest objects for the lifetime of the process",
    )
    skill_cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching git-cloned or downloaded skills. Defaults to ~/.cache/mmar_carl/skills/",
    )

    # Timeout
    timeout: float = Field(default=120.0, description="Timeout in seconds for skill execution")

    # ------------------------------------------------------------------ #
    #  LLM_AGENT mode settings                                            #
    # ------------------------------------------------------------------ #

    llm_max_iterations: int = Field(
        default=8,
        description=(
            "Maximum number of tool-calling iterations in LLM_AGENT mode. "
            "The loop stops when the LLM returns a final response (no tool calls) "
            "or this limit is reached."
        ),
    )

    # ------------------------------------------------------------------ #
    #  Workspace and output settings                                       #
    # ------------------------------------------------------------------ #

    persist_workspace: bool = Field(
        default=False,
        description=(
            "If True, skip workspace cleanup after LLM_AGENT execution. "
            "The workspace path is written to "
            "'$memory.agent_skill.<output_memory_key>_workspace' so downstream steps "
            "can access files produced without copying them. "
            "Only applies to LLM_AGENT mode; other modes do not use a persistent workspace."
        ),
    )

    output_capture: str = Field(
        default="both",
        description=(
            "What to capture as the step result. "
            "'stdout' — LLM/script stdout only; "
            "'files' — list of output files from workspace/out; "
            "'both' — stdout plus file paths."
        ),
    )
    output_files_glob: list[str] = Field(
        default_factory=lambda: ["*"],
        description=(
            "Glob patterns to filter output files collected from workspace/out. "
            "Only applies when output_capture is 'files' or 'both'."
        ),
    )

    # ------------------------------------------------------------------ #
    #  Runtime selection                                                   #
    # ------------------------------------------------------------------ #

    runtime: str = Field(
        default="local",
        description=(
            "Execution runtime for scripts. "
            "'local' — subprocess on host (default, no isolation). "
            "'docker' — requires carl[docker] extra. "
            "'e2b' — requires carl[e2b] extra."
        ),
    )
    runtime_config: dict = Field(
        default_factory=dict,
        description=(
            "Runtime-specific configuration. "
            "docker: {'image': 'ghcr.io/...', 'network': 'none', 'mem_limit': '1g'}. "
            "e2b:    {'template': 'default', 'api_key_env': 'E2B_API_KEY'}."
        ),
    )

    # ------------------------------------------------------------------ #
    #  Trust & integrity                                                   #
    # ------------------------------------------------------------------ #

    trust_policy: str = Field(
        default="any",
        description=(
            "Controls skill integrity verification. "
            "'any' — no verification (default, suitable for trusted/local skills). "
            "'sha_pinned' — skill_sha256 is required and must match the SKILL.md digest."
        ),
    )
    skill_sha256: Optional[str] = Field(
        default=None,
        description=(
            "Expected SHA256 of the skill's SKILL.md file. "
            "Required when trust_policy='sha_pinned'. "
            "For GitHub tarballs, this is the hash of the extracted SKILL.md, "
            "not the tarball itself."
        ),
    )

    # ------------------------------------------------------------------ #
    #  Dependency management                                               #
    # ------------------------------------------------------------------ #

    extra_pip: list[str] = Field(
        default_factory=list,
        description=(
            "Additional Python packages to install before running skill scripts. "
            "Installed into an isolated overlay so the host environment is not modified. "
            "Example: ['pdfplumber>=0.11', 'python-pptx>=1.0']."
        ),
    )

    # ------------------------------------------------------------------ #
    #  Structured output (LLM_AGENT mode only)                             #
    # ------------------------------------------------------------------ #

    output_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional JSON-schema-style dict describing the structure of the "
            "LLM_AGENT final response. When set, the executor: (1) instructs the "
            "agent (in its system context) that the final reply must be JSON "
            "matching this schema; (2) parses the final response as JSON; (3) "
            "validates the parsed value against the schema. On parse/validation "
            "failure the step is marked failed. Only the top-level 'type' "
            "(object/array/string/number/integer/boolean/null), 'properties', "
            "'required', and 'items' keys are honoured — this is a lightweight "
            "checker, not full JSON Schema. Other modes (LLM/SCRIPT/HYBRID/"
            "SUBAGENT) ignore this field."
        ),
    )
    output_schema_strict: bool = Field(
        default=True,
        description=(
            "When True (default), an unparseable or schema-violating LLM_AGENT "
            "final response fails the step. When False, validation failures are "
            "logged as warnings on `result_data['schema_warnings']` but the step "
            "still succeeds — useful when you want best-effort structured output "
            "without blocking the chain."
        ),
    )

    @model_validator(mode="after")
    def coerce_skill_source(self) -> "AgentSkillStepConfig":
        """Auto-coerce plain strings and URIs to AgentSkillSource."""
        if isinstance(self.skill, str):
            s = self.skill

            # github:// URI — use git_url + git_subdirectory
            if s.startswith("github://"):
                from mmar_carl.skill_resolver import _parse_github_uri
                owner, repo, subpath, ref = _parse_github_uri(s)
                git_url = f"https://github.com/{owner}/{repo}"
                self.skill = AgentSkillSource(
                    git_url=git_url,
                    git_ref=ref,
                    git_subdirectory=subpath if subpath else None,
                )
                return self

            # module:// URI — map to package
            if s.startswith("module://"):
                pkg = s.removeprefix("module://")
                self.skill = AgentSkillSource(package=pkg)
                return self

            # local:// URI or plain filesystem paths
            if (
                s.startswith("local://")
                or s.startswith("/")
                or s.startswith("./")
                or s.startswith("../")
                or s.endswith("/")
            ):
                path = s.removeprefix("local://")
                self.skill = AgentSkillSource(path=path)
                return self

            # HTTPS / HTTP tarball URL
            if s.startswith("http://") or s.startswith("https://") or s.startswith("git@"):
                self.skill = AgentSkillSource(git_url=s)
                return self

            # Assume skill name
            self.skill = AgentSkillSource(name=s)
        return self
