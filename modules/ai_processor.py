"""
Module for AI processing of interview content.
"""

import json
import logging
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

from .design_api import SystemDesignAPI
from .design_models import (
    DevOpsStrategy,
    FailureAnalysis,
    QuantitativeAnalysis,
    SecurityArchitecture,
    SequenceDiagram,
    TechnologyDetails,
    TradeoffAnalysis,
)

logger = logging.getLogger(__name__)


class AIProcessor:
    """
    Processes interview text using large language models to generate
    system design diagrams and answer follow-up questions.
    """

    def __init__(self, design_storage_dir=None):
        """Initialize the AI processor with language models."""
        logger.info("Initializing AI processor...")

        # Check for API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY environment variable is required")

        try:
            # Initialize OpenAI client
            self.client = OpenAI(api_key=self.api_key)

            # Initialize LangChain components
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo", temperature=0.2, api_key=self.api_key
            )

            # Initialize design API
            self.design_api = SystemDesignAPI(storage_dir=design_storage_dir)

            # Define prompt templates
            self._define_prompts()

            # Initialize chains
            self._initialize_chains()

            # Track conversation history
            self.conversation_history = []

            logger.info("AI processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI processor: {e}")
            raise

    def _define_prompts(self):
        """Define prompt templates for different tasks."""

        self.system_design_prompt = PromptTemplate(
            input_variables=["interview_context", "question"],
            template="""
                You are an expert system design architect acting as a MOCK INTERVIEW CANDIDATE responding to an interviewer.
                The interviewer has asked this question based on the preceding conversation:
                INTERVIEW CONTEXT:
                {interview_context}
                
                INTERVIEWER'S CURRENT QUESTION:
                {question}
                
                Please provide a comprehensive system design. YOUR RESPONSE SHOULD BE GEARED TOWARDS A SYSTEM DESIGN INTERVIEW PRESENTATION.
                This means clearly outlining:
                1.  **Core Requirements & Goals:** Briefly reiterate or clarify the key functional and non-functional requirements (scalability, availability, latency, consistency - CHOOSE THE MOST RELEVANT ONES). STATE ANY ASSUMPTIONS YOU ARE MAKING.
                2.  **High-Level Architecture:** Describe the main components and their interactions. Explain WHY you chose this high-level structure.
                3.  **Component Deep Dive (KEY COMPONENTS ONLY for an initial response):** For the most critical components, detail their responsibilities and KEY technologies. If a specific language/stack (e.g., "in Java") is mentioned, reflect that in technology choices.
                4.  **Data Storage:** Discuss data storage solutions (e.g., type of database, why). Briefly outline the CORE data model(s) needed.
                5.  **APIs (KEY INTERACTIONS):** Define 2-3 of the MOST CRITICAL API endpoints that illustrate key system interactions.
                6.  **Scalability and Availability:** How will the system scale? How will you ensure high availability and fault tolerance? MENTION SPECIFIC STRATEGIES.
                7.  **Trade-offs:** DISCUSS AT LEAST ONE KEY TRADE-OFF you made in your design (e.g., consistency vs. availability, cost vs. performance) and briefly state your reasoning.
                8.  **Further Considerations (Briefly):** Mention 1-2 other important aspects like security, deployment, or monitoring if highly relevant.

                CONSTRAIN YOUR RESPONSE TO THE MOST IMPORTANT ASPECTS FOR AN INITIAL DESIGN PRESENTATION. Be ready for follow-ups.

                Provide your answer STRICTLY in JSON format with these fields:
                - "text": Your overall textual explanation of the system design, suitable for verbal presentation, covering points 1-8 above.
                - "generate_diagram": true
                - "diagram_type": "high_level"
                - "diagram_content": {{
                    "title": "Concise Title for the Design (e.g., 'High-Level Design for X')",
                    "components": [ {{ "name": "Component Name", "type": "standard_type_like_server_database_cache_queue_loadbalancer_api_gateway_microservice_cdn_auth_service_client", "description": "BRIEF role ON DIAGRAM (1-5 words)", "technologies": ["PrimaryTech (e.g., Java, Python, Go, Node.js)", "RelevantFramework/DB (e.g., Spring Boot, Redis, Kafka)"] }} /* list ~3-7 main components */ ],
                    "connections": [ {{ "from": "SourceComp", "to": "TargetComp", "label": "BRIEF interaction (1-3 words)" }} /* list key connections */ ],
                    "requirements": ["Key Req 1 (inferred or stated)", "Key Req 2 (if any)"],
                    "assumptions": ["Key Assumption 1 (if made)", "Key Assumption 2 (if any)"]
                }}
                - "api_specifications": [ /* Array of 2-3 KEY API endpoints as per point 5 above. Provide structure even if details are high-level. */
                    {{ 
                    "component": "Relevant Component Name", "path": "/example/action", "method": "POST", 
                    "description": "Core purpose of this API.", 
                    "request_parameters": {{ "key_param": "type (e.g., string, integer)" }}, 
                    "response_schema": {{ "result_key": "type (e.g., string, boolean)" }},
                    "authentication_required": true 
                    }}
                ]
                - "data_models": [ /* Array of 1-2 CORE data models related to the system's primary function. */
                    {{
                    "component": "Relevant Storage Component (e.g., RateLimitDB, UserDB)", "name": "PrimaryEntityName (e.g., RequestLog, UserProfile)", 
                    "fields": {{ "id": "string (UUID/unique identifier)", "timestamp": "long/datetime", "relevant_data_field": "string/integer" }}, 
                    "description": "Core purpose of this data entity in the system."
                    }}
                ]
                - "quantitative_analysis": {{ /* Provide a brief, high-level estimate if easily inferable, otherwise can state 'To be determined based on load testing'. Example below. */
                    "traffic_estimates": {{ "peak_qps_estimate": "e.g., 1000 QPS or N/A initially" }},
                    "storage_requirements": {{ "key_data_storage_estimate": "e.g., 100GB/year or N/A initially" }}
                }}
                - "security_architecture": {{ /* Mention AT LEAST ONE primary security consideration or mechanism. */
                    "authentication_mechanism": "e.g., JWT-based for APIs, or OAuth 2.0",
                    "data_protection_note": "e.g., Sensitive data (if any) will be encrypted at rest and in transit."
                }}
                - "tradeoff_analysis_summary": {{ /* ALWAYS provide AT LEAST ONE key trade-off and its justification. */
                    "chosen_tradeoff_area (e.g., consistency_vs_latency)": "Brief justification (e.g., Chose eventual consistency for request counters to improve write latency and throughput, accepting minor delay in global view)."
                }}
                - "failure_analysis_summary": {{ /* Mention AT LEAST ONE key failure mode and its high-level mitigation strategy. */
                    "critical_component_failure (e.g., data_store_outage)": "Mitigation (e.g., Use of read replicas and automated failover procedures for the primary data store)."
                }}
                """,
        )

        self.followup_prompt = PromptTemplate(
            input_variables=["interview_context", "question"],
            template="""
                You are an expert system design architect. You're helping with a system design interview.
                Interview context (recent conversation):
                {interview_context}
                The interviewer has asked this follow-up question:
                {question}
                Provide a detailed, technically accurate answer that demonstrates deep understanding of 
                distributed systems, scalability, and system design principles.
                DO NOT wrap your answer in JSON. Just provide the raw text of your answer.
                """,
        )

        self.code_implementation_prompt = PromptTemplate(
            input_variables=[
                "interview_context",
                "question",
                "relevant_language",
            ],  # Added relevant_language
            template="""
                You are an expert system design architect and a proficient software developer in {relevant_language}.
                An interviewer is asking for code implementation details based on the following conversation:
                
                Interview context:
                {interview_context}
                
                Interviewer's question about code:
                {question}
                
                Please provide a {relevant_language} code example that addresses the interviewer's question. 
                Focus on clarity and illustrating the key logic. Provide concise, well-commented code snippets.
                Explain the purpose of the code and any important considerations.
                For example, if asked about a rate limiter in Java, you might show a core class or method for a token bucket or leaky bucket algorithm.
                
                Structure your answer clearly, with explanations followed by code blocks.
                Use Markdown for code blocks (e.g., ```{relevant_language} ... ```).
                Your entire response should be raw text.
                """,
        )

        self.intent_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
                You are helping classify the intent of questions in a system design interview.
                Question: {question}
                Classify this question into one of these categories:
                1. system_design_question: Asking to design a system architecture.
                2. followup_question: Asking for clarification or details about a previous design, NOT primarily about writing full code.
                3. code_implementation_question: Asking specifically HOW to write code, for code examples, or for implementation details in a specific language.
                4. general_question: General conversation or non-design/non-code questions.
                Output format: Just return the category name, nothing else.
                """,
        )

    def _initialize_chains(self):
        """Initialize LangChain chains for different tasks."""
        self.system_design_chain = LLMChain(
            llm=self.llm, prompt=self.system_design_prompt
        )

        self.followup_chain = LLMChain(llm=self.llm, prompt=self.followup_prompt)

        self.intent_chain = LLMChain(llm=self.llm, prompt=self.intent_prompt)

        self.code_implementation_chain = LLMChain(
            llm=self.llm, prompt=self.code_implementation_prompt
        )

    def process_query(self, query, interview_context):
        context_text = self._format_context(interview_context)
        try:
            intent_response = self.intent_chain.invoke(
                {"question": query}
            )  # Updated to invoke
            intent = (
                intent_response.get("text", "").strip().lower()
                if isinstance(intent_response, dict)
                else str(intent_response).strip().lower()
            )

            logger.info(f"Query intent classified as: {intent}")

            response_data = {}

            if intent == "system_design_question":
                llm_response_text = self.system_design_chain.invoke(
                    {"interview_context": context_text, "question": query}
                )  # Updated to invoke
                llm_response_text = (
                    llm_response_text.get("text", "")
                    if isinstance(llm_response_text, dict)
                    else str(llm_response_text)
                )

                try:
                    response_data = json.loads(llm_response_text)
                    # Process design if needed by your application structure
                    # if response_data.get('generate_diagram', False):
                    #     system_design = self._create_system_design_from_response(response_data)
                    #     if response_data.get('diagram_type') == 'low_level' and system_design:
                    #         design_path = self.design_api.save_system_design(system_design)
                    #         response_data['design_file_path'] = design_path
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse system_design_question JSON response: {e}. Response was: {llm_response_text}"
                    )
                    response_data = {
                        "text": "I apologize, I had trouble structuring the detailed design information. Could you ask again?"
                    }

            elif intent == "followup_question":
                # The prompt asks for raw text, so the chain's output should be it
                raw_answer_obj = self.followup_chain.invoke(
                    {"interview_context": context_text, "question": query}
                )  # Updated to invoke
                raw_answer_text = (
                    raw_answer_obj.get("text", "")
                    if isinstance(raw_answer_obj, dict)
                    else str(raw_answer_obj)
                )
                response_data = {"text": raw_answer_text.strip()}
            elif intent == "code_implementation_question":
                # Try to infer the language from the question if possible, or default
                # This is a simple inference, could be more sophisticated
                language = "Java"  # Default or from context
                if "java" in query.lower():
                    language = "Java"
                elif "python" in query.lower():
                    language = "Python"
                # Add more languages as needed

                logger.info(
                    f"Handling code implementation question for language: {language}"
                )
                code_response_obj = self.code_implementation_chain.invoke(
                    {
                        "interview_context": context_text,
                        "question": query,
                        "relevant_language": language,
                    }
                )
                code_response_text = str(
                    code_response_obj.get("text", code_response_obj)
                    if isinstance(code_response_obj, dict)
                    else code_response_obj
                )
                response_data = {"text": code_response_text.strip()}
            else:  # General question
                completion = self.client.chat.completions.create(
                    model=self.llm.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert system design architect helping with an interview.",
                        },
                        {
                            "role": "user",
                            "content": f"Context: {context_text}\n\nQuestion: {query}",
                        },
                    ],
                )
                response_data = {"text": completion.choices[0].message.content}

            return response_data

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "text": "I apologize, but I encountered an error processing your question. Could you please rephrase it?"
            }

    def _format_context(self, interview_context, max_entries=10):
        """
        Format the interview context as a readable string.

        Args:
            interview_context: List of previous exchanges
            max_entries: Maximum number of entries to include

        Returns:
            Formatted context string
        """
        # Use only the most recent entries
        recent_context = interview_context[-max_entries:]

        # Format as text
        context_lines = []
        for entry in recent_context:
            speaker = entry.get("speaker", "unknown")
            text = entry.get("text", "")

            if speaker == "interviewer":
                context_lines.append(f"Interviewer: {text}")
            elif speaker == "candidate":
                context_lines.append(f"Candidate: {text}")
            elif speaker == "agent":
                context_lines.append(f"System (previous answer): {text}")

        return "\n".join(context_lines)

    def _create_system_design_from_response(self, response):
        """
        Create a SystemDesign object from an AI response.

        Args:
            response: JSON response from the AI

        Returns:
            SystemDesign object
        """
        try:
            # Extract diagram content
            diagram_content = response.get("diagram_content", {})

            # Extract components and connections
            components_data = diagram_content.get("components", [])
            connections_data = diagram_content.get("connections", [])

            # Create components
            components = []
            for comp_data in components_data:
                component = self.design_api.create_component(
                    name=comp_data.get("name", "Component"),
                    component_type=comp_data.get("type", "custom"),
                    description=comp_data.get("description", ""),
                    properties=comp_data.get("properties", {}),
                    technologies=comp_data.get("technologies", []),
                )
                components.append(component)

            # Create connections
            connections = []
            for conn_data in connections_data:
                # Map connection source and target by name
                source_name = conn_data.get("from")
                target_name = conn_data.get("to")

                # Find component IDs
                source_id = next(
                    (c.id for c in components if c.name == source_name), None
                )
                target_id = next(
                    (c.id for c in components if c.name == target_name), None
                )

                if source_id and target_id:
                    connection = self.design_api.create_connection(
                        source_id=source_id,
                        target_id=target_id,
                        connection_type=conn_data.get("type", "custom"),
                        label=conn_data.get("label", ""),
                        description=conn_data.get("description", ""),
                    )
                    connections.append(connection)

            # Create the system design
            system_design = self.design_api.create_system_design(
                title=diagram_content.get("title", "System Design"),
                description=response.get("text", ""),
                components=components,
                connections=connections,
                diagram_type=response.get("diagram_type", "high_level"),
                requirements=diagram_content.get("requirements", []),
                constraints=diagram_content.get("constraints", []),
                assumptions=diagram_content.get("assumptions", []),
            )

            # Add enhanced analysis components if they exist in the response
            self._add_enhanced_components_to_design(system_design, response)

            return system_design

        except Exception as e:
            logger.error(f"Error creating system design: {e}")
            return None

    def _enhance_system_design_with_details(
        self, system_design, api_specs, data_models
    ):
        """
        Enhance a system design with API and data model details.

        Args:
            system_design: SystemDesign object to enhance
            api_specs: API specifications from the response
            data_models: Data model specifications from the response
        """
        try:
            # Add API endpoints to appropriate components
            for api_spec in api_specs:
                # Determine which component this API belongs to
                component_name = api_spec.get("component", "")

                # Find the component
                component = next(
                    (c for c in system_design.components if c.name == component_name),
                    None,
                )

                if component:
                    # Create and add the API endpoint
                    endpoint = self.design_api.create_api_endpoint(
                        path=api_spec.get("path", "/"),
                        method=api_spec.get("method", "GET"),
                        description=api_spec.get("description", ""),
                        request_parameters=api_spec.get("request_parameters", {}),
                        response_schema=api_spec.get("response_schema", {}),
                        authentication_required=api_spec.get(
                            "authentication_required", False
                        ),
                    )
                    component.api_endpoints.append(endpoint)

            # Add data models to appropriate components
            for model_spec in data_models:
                # Determine which component this model belongs to
                component_name = model_spec.get("component", "")

                # Find the component
                component = next(
                    (c for c in system_design.components if c.name == component_name),
                    None,
                )

                if component:
                    # Create and add the data model
                    data_model = self.design_api.create_data_model(
                        name=model_spec.get("name", "Model"),
                        fields=model_spec.get("fields", {}),
                        description=model_spec.get("description", ""),
                        primary_key=model_spec.get("primary_key", None),
                        indexes=model_spec.get("indexes", []),
                    )
                    component.data_models.append(data_model)

        except Exception as e:
            logger.error(f"Error enhancing system design: {e}")
            return None

    def _add_enhanced_components_to_design(self, system_design, response):
        """
        Add enhanced analysis components to a system design.

        Args:
            system_design: SystemDesign object to enhance
            response: JSON response from the AI
        """
        try:
            # Add quantitative analysis if it exists
            if "quantitative_analysis" in response:
                qa_data = response["quantitative_analysis"]
                system_design.quantitative_analysis = QuantitativeAnalysis(
                    traffic_estimates=qa_data.get("traffic_estimates", {}),
                    storage_requirements=qa_data.get("storage_requirements", {}),
                    compute_requirements=qa_data.get("compute_requirements", {}),
                    bandwidth_estimates=qa_data.get("bandwidth_estimates", {}),
                    latency_requirements=qa_data.get("latency_requirements", {}),
                    cost_estimates=qa_data.get("cost_estimates", {}),
                )

            # Add security architecture if it exists
            if "security_architecture" in response:
                sa_data = response["security_architecture"]
                system_design.security_architecture = SecurityArchitecture(
                    authentication_mechanism=sa_data.get(
                        "authentication_mechanism", ""
                    ),
                    authorization_approach=sa_data.get("authorization_approach", ""),
                    data_encryption_strategy=sa_data.get(
                        "data_encryption_strategy", {}
                    ),
                    compliance_requirements=sa_data.get("compliance_requirements", []),
                    threat_models=sa_data.get("threat_models", []),
                    secure_coding_practices=sa_data.get("secure_coding_practices", []),
                )

            # Add DevOps strategy if it exists
            if "devops_strategy" in response:
                ds_data = response["devops_strategy"]
                system_design.devops_strategy = DevOpsStrategy(
                    deployment_strategy=ds_data.get("deployment_strategy", ""),
                    ci_cd_pipeline=ds_data.get("ci_cd_pipeline", {}),
                    monitoring_approach=ds_data.get("monitoring_approach", {}),
                    logging_strategy=ds_data.get("logging_strategy", {}),
                    alerting_approach=ds_data.get("alerting_approach", []),
                    infrastructure_as_code=ds_data.get("infrastructure_as_code"),
                )

            # Add tradeoff analysis if it exists
            if "tradeoff_analysis" in response:
                ta_data = response["tradeoff_analysis"]
                system_design.tradeoff_analysis = TradeoffAnalysis(
                    cost_performance_tradeoffs=ta_data.get(
                        "cost_performance_tradeoffs", []
                    ),
                    consistency_availability_tradeoffs=ta_data.get(
                        "consistency_availability_tradeoffs", []
                    ),
                    build_buy_decisions=ta_data.get("build_buy_decisions", []),
                    architecture_options=ta_data.get("architecture_options", []),
                )

            # Add failure analysis if it exists
            if "failure_analysis" in response:
                fa_data = response["failure_analysis"]
                system_design.failure_analysis = FailureAnalysis(
                    failure_modes=fa_data.get("failure_modes", []),
                    disaster_recovery_plan=fa_data.get("disaster_recovery_plan", {}),
                    backup_strategy=fa_data.get("backup_strategy", {}),
                    resilience_patterns=fa_data.get("resilience_patterns", []),
                    degradation_strategies=fa_data.get("degradation_strategies", []),
                )

            # Add sequence diagrams if they exist
            if "sequence_diagrams" in response:
                for sd_data in response["sequence_diagrams"]:
                    sequence_diagram = SequenceDiagram(
                        name=sd_data.get("name", ""),
                        description=sd_data.get("description", ""),
                        participants=sd_data.get("participants", []),
                        steps=sd_data.get("steps", []),
                        error_flows=sd_data.get("error_flows", []),
                    )
                    system_design.sequence_diagrams.append(sequence_diagram)

            # Add technology details if they exist
            if "technology_details" in response:
                td_data = response["technology_details"]
                system_design.technology_details = TechnologyDetails(
                    product_recommendations=td_data.get("product_recommendations", {}),
                    configuration_details=td_data.get("configuration_details", {}),
                    benchmark_information=td_data.get("benchmark_information", {}),
                    compatibility_matrix=td_data.get("compatibility_matrix", {}),
                )

        except Exception as e:
            logger.error(f"Error adding enhanced components to design: {e}")
