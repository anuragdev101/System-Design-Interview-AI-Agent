"""
Module defining data models and class structures for system design components.
These provide structured representations of common system design elements.
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ComponentType(enum.Enum):
    """Enumeration of common component types in system design."""

    # Computing resources
    SERVER = "server"
    LOAD_BALANCER = "load_balancer"
    APPLICATION_SERVER = "application_server"
    WEB_SERVER = "web_server"
    CACHE = "cache"
    CDN = "cdn"

    # Database types
    RELATIONAL_DB = "relational_db"
    DOCUMENT_DB = "document_db"
    KEY_VALUE_DB = "key_value_db"
    GRAPH_DB = "graph_db"
    COLUMN_DB = "column_db"
    TIME_SERIES_DB = "time_series_db"

    # Messaging and streaming
    MESSAGE_QUEUE = "message_queue"
    EVENT_BUS = "event_bus"
    STREAM_PROCESSOR = "stream_processor"

    # Storage
    BLOB_STORAGE = "blob_storage"
    FILE_STORAGE = "file_storage"
    OBJECT_STORAGE = "object_storage"

    # Services
    API_GATEWAY = "api_gateway"
    SERVICE_MESH = "service_mesh"
    MICROSERVICE = "microservice"
    AUTHENTICATION_SERVICE = "auth_service"

    # Other
    CUSTOM = "custom"


class ConnectionType(enum.Enum):
    """Enumeration of common connection types between components."""

    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MESSAGE = "message"
    STREAM = "stream"
    DATABASE = "database"
    CUSTOM = "custom"


class DiagramType(enum.Enum):
    """Enumeration of diagram types."""

    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"
    SEQUENCE = "sequence"
    COMPONENT = "component"
    DEPLOYMENT = "deployment"
    ENTITY_RELATIONSHIP = "entity_relationship"
    FAILURE_MODE = "failure_mode"
    TRAFFIC_FLOW = "traffic_flow"
    SECURITY_FLOW = "security_flow"


@dataclass
class Component:
    """Data model for a system design component."""

    id: str
    name: str
    component_type: ComponentType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    subcomponents: List["Component"] = field(default_factory=list)

    # Technical details for low-level design
    technologies: List[str] = field(default_factory=list)
    api_endpoints: List["APIEndpoint"] = field(default_factory=list)
    data_models: List["DataModel"] = field(default_factory=list)
    scaling_strategy: Optional[str] = None
    fault_tolerance: Optional[str] = None


@dataclass
class Connection:
    """Data model for a connection between components."""

    source_id: str
    target_id: str
    connection_type: ConnectionType
    label: str = ""
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    # Technical details for low-level design
    protocol: Optional[str] = None
    data_format: Optional[str] = None
    authentication: Optional[str] = None
    rate_limiting: Optional[bool] = None
    is_bidirectional: bool = False


@dataclass
class APIEndpoint:
    """Data model for an API endpoint."""

    path: str
    method: str
    description: str = ""
    request_parameters: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    authentication_required: bool = False
    rate_limited: bool = False


@dataclass
class DataModel:
    """Data model for data entities in the system."""

    name: str
    fields: Dict[str, str]  # Field name -> data type
    description: str = ""
    relationships: List[Dict[str, str]] = field(default_factory=list)
    primary_key: Optional[str] = None
    indexes: List[str] = field(default_factory=list)
    validation_rules: Dict[str, str] = field(
        default_factory=dict
    )  # Field name -> validation rule
    encryption_requirements: Dict[str, str] = field(
        default_factory=dict
    )  # Field name -> encryption requirement
    access_control: Optional[str] = None  # Access control policy for this model


@dataclass
class QuantitativeAnalysis:
    """Data model for quantitative analysis of the system."""

    traffic_estimates: Dict[str, Any] = field(
        default_factory=dict
    )  # QPS, MAU, DAU, etc.
    storage_requirements: Dict[str, Any] = field(default_factory=dict)  # GB, TB, etc.
    compute_requirements: Dict[str, Any] = field(default_factory=dict)  # CPU, memory
    bandwidth_estimates: Dict[str, Any] = field(default_factory=dict)  # Mbps, Gbps
    latency_requirements: Dict[str, Any] = field(default_factory=dict)  # SLAs, ms, etc.
    cost_estimates: Dict[str, Any] = field(default_factory=dict)  # $, $/month, etc.


@dataclass
class SecurityArchitecture:
    """Data model for security aspects of the system."""

    authentication_mechanism: str = ""
    authorization_approach: str = ""
    data_encryption_strategy: Dict[str, str] = field(
        default_factory=dict
    )  # At rest, in transit
    compliance_requirements: List[str] = field(
        default_factory=list
    )  # GDPR, HIPAA, etc.
    threat_models: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Threat models and mitigations
    secure_coding_practices: List[str] = field(default_factory=list)


@dataclass
class DevOpsStrategy:
    """Data model for DevOps and operational aspects."""

    deployment_strategy: str = ""  # Blue-green, canary, etc.
    ci_cd_pipeline: Dict[str, Any] = field(default_factory=dict)
    monitoring_approach: Dict[str, List[str]] = field(
        default_factory=dict
    )  # Metrics to monitor
    logging_strategy: Dict[str, Any] = field(default_factory=dict)
    alerting_approach: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Alert conditions
    infrastructure_as_code: Optional[str] = None  # Approach to IaC


@dataclass
class TradeoffAnalysis:
    """Data model for trade-off analysis."""

    cost_performance_tradeoffs: List[Dict[str, Any]] = field(default_factory=list)
    consistency_availability_tradeoffs: List[Dict[str, Any]] = field(
        default_factory=list
    )
    build_buy_decisions: List[Dict[str, Any]] = field(default_factory=list)
    architecture_options: List[Dict[str, List[str]]] = field(
        default_factory=list
    )  # Options with pros/cons


@dataclass
class FailureAnalysis:
    """Data model for failure scenarios and recovery plans."""

    failure_modes: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Failure modes and impacts
    disaster_recovery_plan: Dict[str, Any] = field(default_factory=dict)
    backup_strategy: Dict[str, Any] = field(default_factory=dict)  # RPO, RTO, etc.
    resilience_patterns: List[str] = field(
        default_factory=list
    )  # Circuit breaker, bulkhead, etc.
    degradation_strategies: List[Dict[str, str]] = field(
        default_factory=list
    )  # Graceful degradation


@dataclass
class SequenceDiagram:
    """Data model for sequence diagrams."""

    name: str
    description: str
    participants: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)  # Sequence of steps
    error_flows: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Error handling sequences


@dataclass
class TechnologyDetails:
    """Data model for technology-specific details."""

    product_recommendations: Dict[str, str] = field(
        default_factory=dict
    )  # Product -> version
    configuration_details: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # Component -> config
    benchmark_information: Dict[str, Any] = field(
        default_factory=dict
    )  # Performance benchmarks
    compatibility_matrix: Dict[str, List[str]] = field(
        default_factory=dict
    )  # Compatible technologies


@dataclass
class SystemDesign:
    """Data model for a complete system design."""

    title: str
    description: str
    components: List[Component]
    connections: List[Connection]
    diagram_type: DiagramType
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    # Additional fields for comprehensive design
    scalability_considerations: Optional[str] = None
    performance_considerations: Optional[str] = None
    security_considerations: Optional[str] = None
    availability_considerations: Optional[str] = None
    cost_considerations: Optional[str] = None

    # Enhanced analysis components
    quantitative_analysis: Optional[QuantitativeAnalysis] = None
    security_architecture: Optional[SecurityArchitecture] = None
    devops_strategy: Optional[DevOpsStrategy] = None
    tradeoff_analysis: Optional[TradeoffAnalysis] = None
    failure_analysis: Optional[FailureAnalysis] = None
    sequence_diagrams: List[SequenceDiagram] = field(default_factory=list)
    technology_details: Optional[TechnologyDetails] = None

    def to_dict(self) -> Dict:
        """Convert the design to a dictionary for serialization."""
        # Implement a proper dictionary conversion that handles nested objects
        design_dict = {
            "title": self.title,
            "description": self.description,
            "diagram_type": self.diagram_type.value,
            "components": [self._component_to_dict(c) for c in self.components],
            "connections": [self._connection_to_dict(c) for c in self.connections],
            "requirements": self.requirements,
            "constraints": self.constraints,
            "assumptions": self.assumptions,
            "scalability_considerations": self.scalability_considerations,
            "performance_considerations": self.performance_considerations,
            "security_considerations": self.security_considerations,
            "availability_considerations": self.availability_considerations,
            "cost_considerations": self.cost_considerations,
        }

        # Add enhanced analysis components if they exist
        if self.quantitative_analysis:
            design_dict["quantitative_analysis"] = {
                "traffic_estimates": self.quantitative_analysis.traffic_estimates,
                "storage_requirements": self.quantitative_analysis.storage_requirements,
                "compute_requirements": self.quantitative_analysis.compute_requirements,
                "bandwidth_estimates": self.quantitative_analysis.bandwidth_estimates,
                "latency_requirements": self.quantitative_analysis.latency_requirements,
                "cost_estimates": self.quantitative_analysis.cost_estimates,
            }

        if self.security_architecture:
            design_dict["security_architecture"] = {
                "authentication_mechanism": self.security_architecture.authentication_mechanism,
                "authorization_approach": self.security_architecture.authorization_approach,
                "data_encryption_strategy": self.security_architecture.data_encryption_strategy,
                "compliance_requirements": self.security_architecture.compliance_requirements,
                "threat_models": self.security_architecture.threat_models,
                "secure_coding_practices": self.security_architecture.secure_coding_practices,
            }

        if self.devops_strategy:
            design_dict["devops_strategy"] = {
                "deployment_strategy": self.devops_strategy.deployment_strategy,
                "ci_cd_pipeline": self.devops_strategy.ci_cd_pipeline,
                "monitoring_approach": self.devops_strategy.monitoring_approach,
                "logging_strategy": self.devops_strategy.logging_strategy,
                "alerting_approach": self.devops_strategy.alerting_approach,
                "infrastructure_as_code": self.devops_strategy.infrastructure_as_code,
            }

        if self.tradeoff_analysis:
            design_dict["tradeoff_analysis"] = {
                "cost_performance_tradeoffs": self.tradeoff_analysis.cost_performance_tradeoffs,
                "consistency_availability_tradeoffs": self.tradeoff_analysis.consistency_availability_tradeoffs,
                "build_buy_decisions": self.tradeoff_analysis.build_buy_decisions,
                "architecture_options": self.tradeoff_analysis.architecture_options,
            }

        if self.failure_analysis:
            design_dict["failure_analysis"] = {
                "failure_modes": self.failure_analysis.failure_modes,
                "disaster_recovery_plan": self.failure_analysis.disaster_recovery_plan,
                "backup_strategy": self.failure_analysis.backup_strategy,
                "resilience_patterns": self.failure_analysis.resilience_patterns,
                "degradation_strategies": self.failure_analysis.degradation_strategies,
            }

        if self.sequence_diagrams:
            design_dict["sequence_diagrams"] = [
                {
                    "name": diagram.name,
                    "description": diagram.description,
                    "participants": diagram.participants,
                    "steps": diagram.steps,
                    "error_flows": diagram.error_flows,
                }
                for diagram in self.sequence_diagrams
            ]

        if self.technology_details:
            design_dict["technology_details"] = {
                "product_recommendations": self.technology_details.product_recommendations,
                "configuration_details": self.technology_details.configuration_details,
                "benchmark_information": self.technology_details.benchmark_information,
                "compatibility_matrix": self.technology_details.compatibility_matrix,
            }

        return design_dict

    def _component_to_dict(self, component: Component) -> Dict:
        """Convert a component to a dictionary for serialization."""
        return {
            "id": component.id,
            "name": component.name,
            "type": component.component_type.value,
            "description": component.description,
            "properties": component.properties,
            "technologies": component.technologies,
            "api_endpoints": [
                {
                    "path": endpoint.path,
                    "method": endpoint.method,
                    "description": endpoint.description,
                    "request_parameters": endpoint.request_parameters,
                    "response_schema": endpoint.response_schema,
                    "authentication_required": endpoint.authentication_required,
                    "rate_limited": endpoint.rate_limited,
                }
                for endpoint in component.api_endpoints
            ],
            "data_models": [
                {
                    "name": model.name,
                    "fields": model.fields,
                    "description": model.description,
                    "relationships": model.relationships,
                    "primary_key": model.primary_key,
                    "indexes": model.indexes,
                }
                for model in component.data_models
            ],
            "scaling_strategy": component.scaling_strategy,
            "fault_tolerance": component.fault_tolerance,
            "subcomponents": [
                self._component_to_dict(sc) for sc in component.subcomponents
            ],
        }

    def _connection_to_dict(self, connection: Connection) -> Dict:
        """Convert a connection to a dictionary for serialization."""
        return {
            "source_id": connection.source_id,
            "target_id": connection.target_id,
            "type": connection.connection_type.value,
            "label": connection.label,
            "description": connection.description,
            "properties": connection.properties,
            "protocol": connection.protocol,
            "data_format": connection.data_format,
            "authentication": connection.authentication,
            "rate_limiting": connection.rate_limiting,
            "is_bidirectional": connection.is_bidirectional,
        }
