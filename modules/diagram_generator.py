"""
Module for generating system design diagrams.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from diagrams.aws.compute import EC2, Lambda
from diagrams.aws.database import RDS, Dynamodb, ElastiCache
from diagrams.aws.network import ELB, CloudFront, Route53
from diagrams.aws.storage import S3
from diagrams.onprem.compute import Server
from diagrams.onprem.database import MongoDB, MySQL, PostgreSQL
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.network import Nginx
from diagrams.onprem.queue import Kafka
from diagrams.programming.framework import Django, Flask, React
from diagrams.programming.language import Go, Java, JavaScript, Python

from diagrams import Cluster, Diagram, Edge

logger = logging.getLogger(__name__)


class DiagramGenerator:
    """
    Generates system design diagrams based on AI-processed interview content.
    Uses the Diagrams library to create architecture diagrams.
    """

    def __init__(self, output_dir=None):
        """
        Initialize the diagram generator.

        Args:
            output_dir: Directory to save generated diagrams
        """
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("diagrams")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Component mappings
        self._initialize_component_mappings()

        logger.info(
            f"Diagram generator initialized with output directory: {self.output_dir}"
        )

    def _initialize_component_mappings(self):
        """Initialize mappings for diagram components."""
        # AWS components
        self.aws_components = {
            "ec2": EC2,
            "lambda": Lambda,
            "rds": RDS,
            "dynamodb": Dynamodb,
            "elasticache": ElastiCache,
            "elb": ELB,
            "route53": Route53,
            "cloudfront": CloudFront,
            "s3": S3,
        }

        # On-premise components
        self.onprem_components = {
            "server": Server,
            "mongodb": MongoDB,
            "mysql": MySQL,
            "postgresql": PostgreSQL,
            "redis": Redis,
            "nginx": Nginx,
            "kafka": Kafka,
        }

        # Programming components
        self.programming_components = {
            "flask": Flask,
            "django": Django,
            "react": React,
            "python": Python,
            "java": Java,
            "go": Go,
            "javascript": JavaScript,
        }

    def generate_diagram(self, diagram_content, diagram_type="high_level"):
        """
        Generate a system design diagram.

        Args:
            diagram_content: Dictionary or JSON string with diagram specifications
            diagram_type: Type of diagram (high_level or low_level)

        Returns:
            Path to the generated diagram file
        """
        try:
            # Parse diagram content if it's a string
            if isinstance(diagram_content, str):
                try:
                    diagram_content = json.loads(diagram_content)
                except json.JSONDecodeError:
                    logger.error("Invalid JSON format in diagram content")
                    return None

            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_design_{diagram_type}_{timestamp}"
            output_path = self.output_dir / filename

            # Generate the diagram
            if diagram_type == "high_level":
                return self._generate_high_level_diagram(diagram_content, output_path)
            elif diagram_type == "low_level":
                return self._generate_low_level_diagram(diagram_content, output_path)
            else:
                logger.warning(f"Unknown diagram type: {diagram_type}")
                return None

        except Exception as e:
            logger.error(f"Error generating diagram: {e}")
            return None

    def _generate_high_level_diagram(self, content, output_path):
        """
        Generate a high-level system design diagram.

        Args:
            content: Dictionary with diagram specifications
            output_path: Output path without extension

        Returns:
            Path to the generated diagram file
        """
        try:
            # Extract components and connections
            components = content.get("components", [])
            connections = content.get("connections", [])

            # Create diagram
            with Diagram(
                name=content.get("title", "System Design"),
                filename=str(output_path),
                outformat="png",
                show=False,
            ) as diagram:
                # Create component nodes
                nodes = {}
                for component in components:
                    component_type = component.get("type", "").lower()
                    component_name = component.get("name", "Component")

                    # Get the appropriate component class
                    component_class = None
                    if component_type in self.aws_components:
                        component_class = self.aws_components[component_type]
                    elif component_type in self.onprem_components:
                        component_class = self.onprem_components[component_type]
                    elif component_type in self.programming_components:
                        component_class = self.programming_components[component_type]
                    else:
                        # Default to Server
                        component_class = Server

                    # Create the node
                    nodes[component_name] = component_class(component_name)

                # Create connections
                for connection in connections:
                    from_node = connection.get("from")
                    to_node = connection.get("to")
                    label = connection.get("label", "")

                    if from_node in nodes and to_node in nodes:
                        if label:
                            nodes[from_node] >> Edge(label=label) >> nodes[to_node]
                        else:
                            nodes[from_node] >> nodes[to_node]

            return f"{output_path}.png"

        except Exception as e:
            logger.error(f"Error generating high-level diagram: {e}")
            return None

    def _generate_low_level_diagram(self, content, output_path):
        """
        Generate a low-level system design diagram with clusters.

        Args:
            content: Dictionary with diagram specifications
            output_path: Output path without extension

        Returns:
            Path to the generated diagram file
        """
        try:
            # Extract components, clusters, and connections
            clusters = content.get("clusters", [])
            connections = content.get("connections", [])

            # Create diagram
            with Diagram(
                name=content.get("title", "Detailed System Design"),
                filename=str(output_path),
                outformat="png",
                show=False,
            ) as diagram:
                # Create nodes dictionary for components
                nodes = {}

                # Create clusters and their components
                for cluster_data in clusters:
                    cluster_name = cluster_data.get("name", "Cluster")
                    components = cluster_data.get("components", [])

                    with Cluster(cluster_name):
                        # Create components in this cluster
                        for component in components:
                            component_type = component.get("type", "").lower()
                            component_name = component.get("name", "Component")

                            # Get the appropriate component class
                            component_class = None
                            if component_type in self.aws_components:
                                component_class = self.aws_components[component_type]
                            elif component_type in self.onprem_components:
                                component_class = self.onprem_components[component_type]
                            elif component_type in self.programming_components:
                                component_class = self.programming_components[
                                    component_type
                                ]
                            else:
                                # Default to Server
                                component_class = Server

                            # Create the node
                            nodes[component_name] = component_class(component_name)

                # Create connections
                for connection in connections:
                    from_node = connection.get("from")
                    to_node = connection.get("to")
                    label = connection.get("label", "")

                    if from_node in nodes and to_node in nodes:
                        if label:
                            nodes[from_node] >> Edge(label=label) >> nodes[to_node]
                        else:
                            nodes[from_node] >> nodes[to_node]

            return f"{output_path}.png"

        except Exception as e:
            logger.error(f"Error generating low-level diagram: {e}")
            return None
