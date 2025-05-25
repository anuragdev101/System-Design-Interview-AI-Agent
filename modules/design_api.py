"""
API module for system design generation and management.
Provides interfaces for creating, retrieving, and manipulating system designs.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .design_models import (
    APIEndpoint,
    Component,
    ComponentType,
    Connection,
    ConnectionType,
    DataModel,
    DiagramType,
    SystemDesign,
)

logger = logging.getLogger(__name__)


class SystemDesignAPI:
    """API for system design generation and management."""

    def __init__(self, storage_dir=None):
        """
        Initialize the system design API.

        Args:
            storage_dir: Directory to store design files
        """
        # Set storage directory
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path("designs")

        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"System Design API initialized with storage directory: {self.storage_dir}"
        )

    def create_component(
        self,
        name: str,
        component_type: Union[str, ComponentType],
        description: str = "",
        properties: Dict[str, Any] = None,
        technologies: List[str] = None,
    ) -> Component:
        """
        Create a new component.

        Args:
            name: Name of the component
            component_type: Type of the component
            description: Description of the component
            properties: Additional properties of the component
            technologies: Technologies used by the component

        Returns:
            The created component
        """
        # Convert component_type to enum if it's a string
        if isinstance(component_type, str):
            try:
                component_type = ComponentType(component_type)
            except ValueError:
                component_type = ComponentType.CUSTOM

        # Create component
        component = Component(
            id=str(uuid.uuid4()),
            name=name,
            component_type=component_type,
            description=description,
            properties=properties or {},
            technologies=technologies or [],
        )

        logger.info(f"Created component: {component.name} ({component.id})")

        return component

    def create_connection(
        self,
        source_id: str,
        target_id: str,
        connection_type: Union[str, ConnectionType],
        label: str = "",
        description: str = "",
        protocol: str = None,
    ) -> Connection:
        """
        Create a new connection between components.

        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            connection_type: Type of the connection
            label: Label for the connection
            description: Description of the connection
            protocol: Protocol used by the connection

        Returns:
            The created connection
        """
        # Convert connection_type to enum if it's a string
        if isinstance(connection_type, str):
            try:
                connection_type = ConnectionType(connection_type)
            except ValueError:
                connection_type = ConnectionType.CUSTOM

        # Create connection
        connection = Connection(
            source_id=source_id,
            target_id=target_id,
            connection_type=connection_type,
            label=label,
            description=description,
            protocol=protocol,
        )

        logger.info(f"Created connection from {source_id} to {target_id}")

        return connection

    def create_api_endpoint(
        self,
        path: str,
        method: str,
        description: str = "",
        request_parameters: Dict[str, Any] = None,
        response_schema: Dict[str, Any] = None,
        authentication_required: bool = False,
    ) -> APIEndpoint:
        """
        Create a new API endpoint.

        Args:
            path: Path of the endpoint
            method: HTTP method
            description: Description of the endpoint
            request_parameters: Parameters for the request
            response_schema: Schema for the response
            authentication_required: Whether authentication is required

        Returns:
            The created API endpoint
        """
        # Create API endpoint
        endpoint = APIEndpoint(
            path=path,
            method=method,
            description=description,
            request_parameters=request_parameters or {},
            response_schema=response_schema or {},
            authentication_required=authentication_required,
        )

        logger.info(f"Created API endpoint: {method} {path}")

        return endpoint

    def create_data_model(
        self,
        name: str,
        fields: Dict[str, str],
        description: str = "",
        primary_key: str = None,
        indexes: List[str] = None,
    ) -> DataModel:
        """
        Create a new data model.

        Args:
            name: Name of the data model
            fields: Fields of the data model
            description: Description of the data model
            primary_key: Primary key field
            indexes: Indexed fields

        Returns:
            The created data model
        """
        # Create data model
        data_model = DataModel(
            name=name,
            fields=fields,
            description=description,
            primary_key=primary_key,
            indexes=indexes or [],
        )

        logger.info(f"Created data model: {name}")

        return data_model

    def create_system_design(
        self,
        title: str,
        description: str,
        components: List[Component],
        connections: List[Connection],
        diagram_type: Union[str, DiagramType],
        requirements: List[str] = None,
        constraints: List[str] = None,
        assumptions: List[str] = None,
    ) -> SystemDesign:
        """
        Create a new system design.

        Args:
            title: Title of the design
            description: Description of the design
            components: Components in the design
            connections: Connections in the design
            diagram_type: Type of diagram
            requirements: Requirements for the design
            constraints: Constraints for the design
            assumptions: Assumptions for the design

        Returns:
            The created system design
        """
        # Convert diagram_type to enum if it's a string
        if isinstance(diagram_type, str):
            try:
                diagram_type = DiagramType(diagram_type)
            except ValueError:
                diagram_type = DiagramType.HIGH_LEVEL

        # Create system design
        system_design = SystemDesign(
            title=title,
            description=description,
            components=components,
            connections=connections,
            diagram_type=diagram_type,
            requirements=requirements or [],
            constraints=constraints or [],
            assumptions=assumptions or [],
        )

        logger.info(f"Created system design: {title}")

        return system_design

    def save_system_design(self, design: SystemDesign) -> str:
        """
        Save a system design to a file.

        Args:
            design: The system design to save

        Returns:
            Path to the saved file
        """
        # Generate a filename
        filename = (
            f"{design.title.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.json"
        )
        file_path = self.storage_dir / filename

        # Convert design to dictionary
        design_dict = design.to_dict()

        # Save to file
        with open(file_path, "w") as f:
            json.dump(design_dict, f, indent=2)

        logger.info(f"Saved system design to: {file_path}")

        return str(file_path)

    def load_system_design(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a system design from a file.

        Args:
            file_path: Path to the design file

        Returns:
            The loaded design as a dictionary, or None if loading fails
        """
        try:
            # Load from file
            with open(file_path, "r") as f:
                design_dict = json.load(f)

            logger.info(f"Loaded system design from: {file_path}")

            return design_dict

        except Exception as e:
            logger.error(f"Error loading system design from {file_path}: {e}")
            return None

    def translate_high_to_low_level(
        self, high_level_design: SystemDesign
    ) -> SystemDesign:
        """
        Translate a high-level design to a low-level design.

        Args:
            high_level_design: The high-level design

        Returns:
            The corresponding low-level design
        """
        # Create a new low-level design
        low_level_design = SystemDesign(
            title=f"{high_level_design.title} - Low Level",
            description=high_level_design.description,
            components=[],
            connections=[],
            diagram_type=DiagramType.LOW_LEVEL,
            requirements=high_level_design.requirements,
            constraints=high_level_design.constraints,
            assumptions=high_level_design.assumptions,
        )

        # ID mapping from high-level to low-level
        id_mapping = {}

        # Add detailed components based on high-level components
        for high_level_component in high_level_design.components:
            # Create corresponding low-level components
            low_level_components = self._expand_to_low_level_components(
                high_level_component
            )

            # Store ID mapping for the main component
            if low_level_components:
                id_mapping[high_level_component.id] = low_level_components[0].id

            # Add components to the design
            low_level_design.components.extend(low_level_components)

        # Add connections based on high-level connections
        for high_level_connection in high_level_design.connections:
            if (
                high_level_connection.source_id in id_mapping
                and high_level_connection.target_id in id_mapping
            ):
                # Create corresponding low-level connection
                low_level_connection = Connection(
                    source_id=id_mapping[high_level_connection.source_id],
                    target_id=id_mapping[high_level_connection.target_id],
                    connection_type=high_level_connection.connection_type,
                    label=high_level_connection.label,
                    description=high_level_connection.description,
                    protocol=self._determine_protocol(high_level_connection),
                )

                low_level_design.connections.append(low_level_connection)

        logger.info(
            f"Translated high-level design '{high_level_design.title}' to low-level design"
        )

        return low_level_design

    def _expand_to_low_level_components(self, component: Component) -> List[Component]:
        """
        Expand a high-level component to low-level components.

        Args:
            component: The high-level component

        Returns:
            List of low-level components
        """
        # This is a simplified implementation
        # In a real-world scenario, this would be more sophisticated
        # and expand components based on their type and requirements

        # Create a more detailed version of the component
        detailed_component = Component(
            id=str(uuid.uuid4()),
            name=component.name,
            component_type=component.component_type,
            description=component.description,
            properties=component.properties.copy(),
            technologies=self._suggest_technologies(component),
        )

        # Add API endpoints based on component type
        detailed_component.api_endpoints = self._suggest_api_endpoints(component)

        # Add data models based on component type
        detailed_component.data_models = self._suggest_data_models(component)

        # Add scaling strategy and fault tolerance
        detailed_component.scaling_strategy = self._suggest_scaling_strategy(component)
        detailed_component.fault_tolerance = self._suggest_fault_tolerance(component)

        return [detailed_component]

    def _suggest_technologies(self, component: Component) -> List[str]:
        """Suggest technologies based on component type."""
        technologies = []

        # Suggest technologies based on component type
        if component.component_type == ComponentType.WEB_SERVER:
            technologies = ["Nginx", "Apache", "NodeJS"]
        elif component.component_type == ComponentType.APPLICATION_SERVER:
            technologies = ["Spring Boot", "Django", "Express", "Flask"]
        elif component.component_type == ComponentType.RELATIONAL_DB:
            technologies = ["PostgreSQL", "MySQL", "Oracle", "SQL Server"]
        elif component.component_type == ComponentType.DOCUMENT_DB:
            technologies = ["MongoDB", "Couchbase", "Firestore"]
        elif component.component_type == ComponentType.KEY_VALUE_DB:
            technologies = ["Redis", "DynamoDB", "etcd"]
        elif component.component_type == ComponentType.CACHE:
            technologies = ["Redis", "Memcached"]
        elif component.component_type == ComponentType.MESSAGE_QUEUE:
            technologies = ["Kafka", "RabbitMQ", "ActiveMQ"]
        elif component.component_type == ComponentType.LOAD_BALANCER:
            technologies = ["HAProxy", "Nginx", "ELB"]

        return technologies

    def _suggest_api_endpoints(self, component: Component) -> List[APIEndpoint]:
        """Suggest API endpoints based on component type."""
        endpoints = []

        # Suggest endpoints based on component type
        if component.component_type in [
            ComponentType.WEB_SERVER,
            ComponentType.APPLICATION_SERVER,
            ComponentType.API_GATEWAY,
        ]:
            endpoints = [
                APIEndpoint(
                    path="/api/health",
                    method="GET",
                    description="Health check endpoint",
                    request_parameters={},
                    response_schema={"status": "string"},
                ),
                APIEndpoint(
                    path="/api/v1/resources",
                    method="GET",
                    description="List all resources",
                    request_parameters={"page": "integer", "limit": "integer"},
                    response_schema={"items": "array", "total": "integer"},
                ),
                APIEndpoint(
                    path="/api/v1/resources/{id}",
                    method="GET",
                    description="Get a specific resource",
                    request_parameters={"id": "string"},
                    response_schema={
                        "id": "string",
                        "name": "string",
                        "properties": "object",
                    },
                ),
            ]

        return endpoints

    def _suggest_data_models(self, component: Component) -> List[DataModel]:
        """Suggest data models based on component type."""
        data_models = []

        # Suggest data models based on component type
        if component.component_type in [
            ComponentType.RELATIONAL_DB,
            ComponentType.DOCUMENT_DB,
            ComponentType.KEY_VALUE_DB,
        ]:
            data_models = [
                DataModel(
                    name="User",
                    fields={
                        "id": "string",
                        "username": "string",
                        "email": "string",
                        "created_at": "timestamp",
                    },
                    description="User account information",
                    primary_key="id",
                    indexes=["email", "username"],
                ),
                DataModel(
                    name="Resource",
                    fields={
                        "id": "string",
                        "name": "string",
                        "description": "string",
                        "owner_id": "string",
                        "created_at": "timestamp",
                        "updated_at": "timestamp",
                    },
                    description="Resource information",
                    primary_key="id",
                    indexes=["owner_id"],
                ),
            ]

        return data_models

    def _suggest_scaling_strategy(self, component: Component) -> str:
        """Suggest scaling strategy based on component type."""
        if component.component_type in [
            ComponentType.WEB_SERVER,
            ComponentType.APPLICATION_SERVER,
        ]:
            return "Horizontal scaling with auto-scaling based on CPU utilization and request count"
        elif component.component_type in [ComponentType.RELATIONAL_DB]:
            return "Vertical scaling for primary, read replicas for scaling reads"
        elif component.component_type in [
            ComponentType.DOCUMENT_DB,
            ComponentType.KEY_VALUE_DB,
        ]:
            return "Horizontal scaling with sharding based on key ranges"
        elif component.component_type in [ComponentType.CACHE]:
            return "Cluster with consistent hashing for distribution"
        else:
            return "Appropriate scaling based on load patterns"

    def _suggest_fault_tolerance(self, component: Component) -> str:
        """Suggest fault tolerance strategy based on component type."""
        if component.component_type in [
            ComponentType.WEB_SERVER,
            ComponentType.APPLICATION_SERVER,
        ]:
            return "Multiple instances across availability zones with health checks"
        elif component.component_type in [ComponentType.RELATIONAL_DB]:
            return "Primary-replica setup with automated failover"
        elif component.component_type in [
            ComponentType.DOCUMENT_DB,
            ComponentType.KEY_VALUE_DB,
        ]:
            return "Distributed replicas with quorum-based consistency"
        elif component.component_type in [ComponentType.CACHE]:
            return "In-memory replication with persistence to disk"
        else:
            return "Redundancy and failover mechanisms"

    def _determine_protocol(self, connection: Connection) -> str:
        """Determine protocol based on connection type."""
        if connection.connection_type == ConnectionType.HTTP:
            return "HTTP/1.1"
        elif connection.connection_type == ConnectionType.HTTPS:
            return "HTTP/1.1 over TLS"
        elif connection.connection_type == ConnectionType.REST:
            return "HTTP/1.1 or HTTP/2 with REST semantics"
        elif connection.connection_type == ConnectionType.GRAPHQL:
            return "HTTP with GraphQL"
        elif connection.connection_type == ConnectionType.GRPC:
            return "HTTP/2 with Protocol Buffers"
        elif connection.connection_type == ConnectionType.WEBSOCKET:
            return "WebSocket (RFC 6455)"
        elif connection.connection_type == ConnectionType.MESSAGE:
            return "AMQP or Kafka Protocol"
        elif connection.connection_type == ConnectionType.TCP:
            return "TCP"
        elif connection.connection_type == ConnectionType.UDP:
            return "UDP"
        else:
            return "Appropriate protocol for the connection type"
