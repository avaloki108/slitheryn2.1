import logging
from typing import Union, List, Type, Dict, Optional

from crytic_compile import CryticCompile, InvalidCompilation

# pylint: disable= no-name-in-module
from slitheryn.core.compilation_unit import SlitherCompilationUnit
from slitheryn.core.slither_core import SlitherCore
from slitheryn.detectors.abstract_detector import AbstractDetector, DetectorClassification
from slitheryn.exceptions import SlitherError
from slitheryn.printers.abstract_printer import AbstractPrinter
from slitheryn.solc_parsing.slither_compilation_unit_solc import SlitherCompilationUnitSolc
from slitheryn.vyper_parsing.vyper_compilation_unit import VyperCompilationUnit
from slitheryn.utils.output import Output
from slitheryn.ai.embedding_service import EmbeddingService
from slitheryn.ai.vector_store import VectorStore
from slitheryn.vyper_parsing.ast.ast import parse

logger = logging.getLogger("Slitheryn")
logging.basicConfig()

logger_detector = logging.getLogger("Detectors")
logger_printer = logging.getLogger("Printers")


def _check_common_things(
    thing_name: str, cls: Type, base_cls: Type, instances_list: List[Type[AbstractDetector]]
) -> None:

    if not issubclass(cls, base_cls) or cls is base_cls:
        raise SlitherError(
            f"You can't register {cls!r} as a {thing_name}. You need to pass a class that inherits from {base_cls.__name__}"
        )

    if any(type(obj) == cls for obj in instances_list):  # pylint: disable=unidiomatic-typecheck
        raise SlitherError(f"You can't register {cls!r} twice.")


def _update_file_scopes(
    sol_parser: SlitherCompilationUnitSolc,
):  # pylint: disable=too-many-branches
    """
    Since all definitions in a file are exported by default, including definitions from its (transitive) dependencies,
    we can identify all top level items that could possibly be referenced within the file from its exportedSymbols.
    It is not as straightforward for user defined types and functions as well as aliasing. See add_accessible_scopes for more details.
    """
    candidates = sol_parser.compilation_unit.scopes.values()
    learned_something = False
    # Because solc's import allows cycle in the import graph, iterate until we aren't adding new information to the scope.
    while True:
        for candidate in candidates:
            learned_something |= candidate.add_accessible_scopes()
        if not learned_something:
            break
        learned_something = False

    for scope in candidates:
        for refId in scope.exported_symbols:
            if refId in sol_parser.contracts_by_id:
                contract = sol_parser.contracts_by_id[refId]
                scope.contracts[contract.name] = contract
            elif refId in sol_parser.functions_by_id:
                functions = sol_parser.functions_by_id[refId]
                assert len(functions) == 1
                scope.functions.add(functions[0])
            elif refId in sol_parser.imports_by_id:
                import_directive = sol_parser.imports_by_id[refId]
                scope.imports.add(import_directive)
            elif refId in sol_parser.top_level_variables_by_id:
                top_level_variable = sol_parser.top_level_variables_by_id[refId]
                scope.variables[top_level_variable.name] = top_level_variable
            elif refId in sol_parser.top_level_events_by_id:
                top_level_event = sol_parser.top_level_events_by_id[refId]
                scope.events.add(top_level_event)
            elif refId in sol_parser.top_level_structures_by_id:
                top_level_struct = sol_parser.top_level_structures_by_id[refId]
                scope.structures[top_level_struct.name] = top_level_struct
            elif refId in sol_parser.top_level_type_aliases_by_id:
                top_level_type_alias = sol_parser.top_level_type_aliases_by_id[refId]
                scope.type_aliases[top_level_type_alias.name] = top_level_type_alias
            elif refId in sol_parser.top_level_enums_by_id:
                top_level_enum = sol_parser.top_level_enums_by_id[refId]
                scope.enums[top_level_enum.name] = top_level_enum
            elif refId in sol_parser.top_level_errors_by_id:
                top_level_custom_error = sol_parser.top_level_errors_by_id[refId]
                scope.custom_errors.add(top_level_custom_error)
            else:
                logger.error(
                    f"Failed to resolved name for reference id {refId} in {scope.filename.absolute}."
                )


class Slither(
    SlitherCore
):  # pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements,too-many-branches
    def __init__(self, target: Union[str, CryticCompile], **kwargs) -> None:
        """
        Args:
            target (str | CryticCompile)
        Keyword Args:
            solc (str): solc binary location (default 'solc')
            disable_solc_warnings (bool): True to disable solc warnings (default false)
            solc_args (str): solc arguments (default '')
            ast_format (str): ast format (default '--ast-compact-json')
            filter_paths (list(str)): list of path to filter (default [])
            triage_mode (bool): if true, switch to triage mode (default false)
            exclude_dependencies (bool): if true, exclude results that are only related to dependencies
            generate_patches (bool): if true, patches are generated (json output only)
            change_line_prefix (str): Change the line prefix (default #)
                for the displayed source codes (i.e. file.sol#1).

        """
        super().__init__()

        self._disallow_partial: bool = kwargs.get("disallow_partial", False)
        self._skip_assembly: bool = kwargs.get("skip_assembly", False)
        self._show_ignored_findings: bool = kwargs.get("show_ignored_findings", False)

        self.line_prefix = kwargs.get("change_line_prefix", "#")

        # Indicate if Codex related features should be used
        self.codex_enabled = kwargs.get("codex", False)
        self.codex_contracts = kwargs.get("codex_contracts", "all")
        self.codex_model = kwargs.get("codex_model", "text-davinci-003")
        self.codex_temperature = kwargs.get("codex_temperature", 0)
        self.codex_max_tokens = kwargs.get("codex_max_tokens", 300)
        self.codex_log = kwargs.get("codex_log", False)
        self.codex_organization: Optional[str] = kwargs.get("codex_organization", None)

        self.no_fail = kwargs.get("no_fail", False)

        self._parsers: List[SlitherCompilationUnitSolc] = []
        try:
            if isinstance(target, CryticCompile):
                crytic_compile = target
            else:
                crytic_compile = CryticCompile(target, **kwargs)
            self._crytic_compile = crytic_compile
        except InvalidCompilation as e:
            # pylint: disable=raise-missing-from
            raise SlitherError(f"Invalid compilation: \n{str(e)}")
        for compilation_unit in crytic_compile.compilation_units.values():
            compilation_unit_slither = SlitherCompilationUnit(self, compilation_unit)
            self._compilation_units.append(compilation_unit_slither)

            if compilation_unit_slither.is_vyper:
                vyper_parser = VyperCompilationUnit(compilation_unit_slither)
                for path, ast in compilation_unit.asts.items():
                    ast_nodes = parse(ast["ast"])
                    vyper_parser.parse_module(ast_nodes, path)
                self._parsers.append(vyper_parser)
            else:
                # Solidity specific
                assert compilation_unit_slither.is_solidity
                sol_parser = SlitherCompilationUnitSolc(compilation_unit_slither)
                self._parsers.append(sol_parser)
                for path, ast in compilation_unit.asts.items():
                    sol_parser.parse_top_level_items(ast, path)
                    self.add_source_code(path)

                for contract in sol_parser._underlying_contract_to_parser:
                    if contract.name.startswith("SlitherynInternalTopLevelContract"):
                        raise SlitherError(
                            # region multi-line-string
                            """Your codebase has a contract named 'SlitherynInternalTopLevelContract'.
        Please rename it, this name is reserved for Slitheryn's internals"""
                            # endregion multi-line
                        )
                    sol_parser._contracts_by_id[contract.id] = contract
                    sol_parser._compilation_unit.contracts.append(contract)

                _update_file_scopes(sol_parser)

        if kwargs.get("generate_patches", False):
            self.generate_patches = True

        self._markdown_root = kwargs.get("markdown_root", "")

        self._detectors = []
        self._printers = []

        filter_paths = kwargs.get("filter_paths", [])
        for p in filter_paths:
            self.add_path_to_filter(p)

        include_paths = kwargs.get("include_paths", [])
        for p in include_paths:
            self.add_path_to_include(p)

        self._exclude_dependencies = kwargs.get("exclude_dependencies", False)

        triage_mode = kwargs.get("triage_mode", False)
        triage_database = kwargs.get("triage_database", "slitheryn.db.json")
        self._triage_mode = triage_mode
        self._previous_results_filename = triage_database

        printers_to_run = kwargs.get("printers_to_run", "")
        if printers_to_run == "echidna":
            self.skip_data_dependency = True

        # Used in inheritance-graph printer
        self.include_interfaces = kwargs.get("include_interfaces", False)

        # RAG embeddings (optional)
        self._enable_rag = kwargs.get("enable_rag", False)
        self._embedding_service: Optional[EmbeddingService] = None
        self._vector_store: Optional[VectorStore] = None
        self._embedding_cache_path: Optional[str] = None

        self._init_rag_configuration()

        self._init_parsing_and_analyses(kwargs.get("skip_analyze", False))

    def _init_parsing_and_analyses(self, skip_analyze: bool) -> None:
        for parser in self._parsers:
            try:
                parser.parse_contracts()
            except Exception as e:
                if self.no_fail:
                    continue
                raise e

        # Embed contracts for RAG before analysis (optional)
        if self._enable_rag:
            self._embed_all_contracts()

        # skip_analyze is only used for testing
        if not skip_analyze:
            for parser in self._parsers:
                try:
                    parser.analyze_contracts()
                except Exception as e:
                    if self.no_fail:
                        continue
                    raise e

    # RAG helpers ----------------------------------------------------------
    def _init_rag_configuration(self) -> None:
        """
        Configure RAG from AI config if available.
        Supports multiple embedding providers: 'mixedbread' (default) or 'ollama'.
        """
        self._ollama_url = "http://localhost:11434"
        self._embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
        self._embedding_provider = "mixedbread"
        self._cache_embeddings = True
        self._embedding_cache_path = ".slitheryn/embeddings_cache/embeddings.json"
        self._ai_config = None

        try:
            from slitheryn.ai.config import get_ai_config  # type: ignore

            self._ai_config = get_ai_config()
            self._ollama_url = self._ai_config.get_ollama_url()
            cfg = self._ai_config.config
            self._enable_rag = self._enable_rag or getattr(cfg, "enable_rag", False)
            self._embedding_provider = getattr(cfg, "embedding_provider", self._embedding_provider)
            self._embedding_model = getattr(cfg, "embedding_model", self._embedding_model)
            self._cache_embeddings = getattr(cfg, "cache_embeddings", self._cache_embeddings)
            self._embedding_cache_path = getattr(
                cfg, "cache_path", self._embedding_cache_path
            )
            self._similarity_threshold = getattr(cfg, "similarity_threshold", 0.7)
            self._max_similar_contracts = getattr(cfg, "max_similar_contracts", 3)
        except Exception as e:
            # Best-effort config; fall back to defaults
            logger.debug(f"Could not load AI config: {e}")
            self._similarity_threshold = 0.7
            self._max_similar_contracts = 3

        if self._enable_rag:
            # Use the config manager to create the appropriate embedding service
            if self._ai_config:
                self._embedding_service = self._ai_config.create_embedding_service()
                logger.info(f"Using {self._embedding_provider} embedding provider with model {self._embedding_model}")
            else:
                # Fallback to Ollama if no config manager
                self._embedding_service = EmbeddingService(
                    base_url=self._ollama_url,
                    model=self._embedding_model,
                    timeout=120,
                )
            self._vector_store = VectorStore()
            if self._cache_embeddings and self._embedding_cache_path:
                try:
                    self._vector_store.load_from_cache(self._embedding_cache_path)
                except Exception:
                    pass

    def _embed_all_contracts(self) -> None:
        """
        Embed all parsed contracts and store in vector store.
        """
        if not self._embedding_service or not self._vector_store:
            return
        if not self._embedding_service.check_model_availability():
            return

        for compilation_unit in self.compilation_units:
            for contract in compilation_unit.contracts:
                if getattr(contract, "is_interface", False):
                    continue
                if contract.name in getattr(self._vector_store, "_store", {}):
                    continue
                source = self._read_contract_source(contract)
                if not source:
                    continue
                metadata = {
                    "file_path": str(getattr(contract.source_mapping.filename, "absolute", "")),
                    "name": contract.name,
                    "code": source,
                    "code_snippet": source[:2000],  # keep snippet for prompt context
                }
                embedding = self._embedding_service.embed_contract(source, contract.name)
                if embedding:
                    self._vector_store.add_contract(contract.name, embedding, metadata)

        if self._cache_embeddings and self._embedding_cache_path:
            try:
                self._vector_store.save_to_cache(self._embedding_cache_path)
            except Exception:
                pass

    def _read_contract_source(self, contract) -> Optional[str]:
        try:
            if hasattr(contract, "source_mapping") and contract.source_mapping:
                mapping = contract.source_mapping
                if hasattr(mapping, "filename") and mapping.filename:
                    with open(mapping.filename.absolute, "r", encoding="utf-8") as f:
                        return f.read()
        except Exception:
            return None
        return None

    @property
    def detectors(self):
        return self._detectors

    @property
    def detectors_high(self):
        return [d for d in self.detectors if d.IMPACT == DetectorClassification.HIGH]

    @property
    def detectors_medium(self):
        return [d for d in self.detectors if d.IMPACT == DetectorClassification.MEDIUM]

    @property
    def detectors_low(self):
        return [d for d in self.detectors if d.IMPACT == DetectorClassification.LOW]

    @property
    def detectors_informational(self):
        return [d for d in self.detectors if d.IMPACT == DetectorClassification.INFORMATIONAL]

    @property
    def detectors_optimization(self):
        return [d for d in self.detectors if d.IMPACT == DetectorClassification.OPTIMIZATION]

    def register_detector(self, detector_class: Type[AbstractDetector]) -> None:
        """
        :param detector_class: Class inheriting from `AbstractDetector`.
        """
        _check_common_things("detector", detector_class, AbstractDetector, self._detectors)

        for compilation_unit in self.compilation_units:
            instance = detector_class(compilation_unit, self, logger_detector)
            self._detectors.append(instance)

    def unregister_detector(self, detector_class: Type[AbstractDetector]) -> None:
        """
        :param detector_class: Class inheriting from `AbstractDetector`.
        """

        for obj in self._detectors:
            if isinstance(obj, detector_class):
                self._detectors.remove(obj)
                return

    def register_printer(self, printer_class: Type[AbstractPrinter]) -> None:
        """
        :param printer_class: Class inheriting from `AbstractPrinter`.
        """
        _check_common_things("printer", printer_class, AbstractPrinter, self._printers)

        instance = printer_class(self, logger_printer)
        self._printers.append(instance)

    def unregister_printer(self, printer_class: Type[AbstractPrinter]) -> None:
        """
        :param printer_class: Class inheriting from `AbstractPrinter`.
        """

        for obj in self._printers:
            if isinstance(obj, printer_class):
                self._printers.remove(obj)
                return

    def run_detectors(self) -> List[Dict]:
        """
        :return: List of registered detectors results.
        """

        self.load_previous_results()
        results = [d.detect() for d in self._detectors]

        self.write_results_to_hide()
        return results

    def run_printers(self) -> List[Output]:
        """
        :return: List of registered printers outputs.
        """

        return [p.output(self._crytic_compile.target).data for p in self._printers]

    @property
    def triage_mode(self) -> bool:
        return self._triage_mode
