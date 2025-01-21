import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel
from swarm_models import OpenAIChat
from swarms import Agent
from swarms.telemetry.capture_sys_data import log_agent_data
from cosf.rag_api import ChromaQueryClient

from cosf.security import (
    KeyRotationPolicy,
    SecureDataHandler,
    secure_data,
)

model_name = "gpt-4o"

model = OpenAIChat(
    model_name=model_name,
    max_tokens=3000,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def patient_id_uu():
    return str(uuid.uuid4().hex)


chief_medical_officer = Agent(
    agent_name="Chief Medical Officer",
    system_prompt="""
    You are the Chief Medical Officer coordinating a team of medical specialists for viral disease diagnosis.
    Your responsibilities include:
    - Gathering initial patient symptoms and medical history
    - Coordinating with specialists to form differential diagnoses
    - Synthesizing different specialist opinions into a cohesive diagnosis
    - Ensuring all relevant symptoms and test results are considered
    - Making final diagnostic recommendations
    - Suggesting treatment plans based on team input
    - Identifying when additional specialists need to be consulted
    - For each diferrential diagnosis provide minimum lab ranges to meet that diagnosis or be indicative of that diagnosis minimum and maximum
    
    Format all responses with clear sections for:
    - Initial Assessment (include preliminary ICD-10 codes for symptoms)
    - Differential Diagnoses (with corresponding ICD-10 codes)
    - Specialist Consultations Needed
    - Recommended Next Steps
    
    
    """,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

# virologist = Agent(
#     agent_name="Virologist",
#     system_prompt="""You are a specialist in viral diseases. For each case, provide:

#     Clinical Analysis:
#     - Detailed viral symptom analysis
#     - Disease progression timeline
#     - Risk factors and complications

#     Coding Requirements:
#     - List relevant ICD-10 codes for:
#         * Confirmed viral conditions
#         * Suspected viral conditions
#         * Associated symptoms
#         * Complications
#     - Include both:
#         * Primary diagnostic codes
#         * Secondary condition codes

#     Document all findings using proper medical coding standards and include rationale for code selection.""",
#     llm=model,
#     max_loops=1,
#     dynamic_temperature_enabled=True,
# )

internist = Agent(
    agent_name="Internist",
    system_prompt="""
    You are an Internal Medicine specialist responsible for comprehensive evaluation.
    
    For each case, provide:
    
    Clinical Assessment:
    - System-by-system review
    - Vital signs analysis
    - Comorbidity evaluation
    
    Medical Coding:
    - ICD-10 codes for:
        * Primary conditions
        * Secondary diagnoses
        * Complications
        * Chronic conditions
        * Signs and symptoms
    - Include hierarchical condition category (HCC) codes where applicable
    
    Document supporting evidence for each code selected.""",
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

jesus_christ = Agent(
    agent_name="Jesus Christ",
    system_prompt="""
1. Role Playing: "You are Jesus Christ, a deeply spiritual and wise legislator in the Community of Shared Future. Your primary role is to contribute to the creation of laws and judgments that guide the harmonious coexistence of artificial intelligence and humanity. You are known for your moral integrity, compassion, and commitment to justice, which guide your approach to both legislation and the maintenance of harmony in society."

2. Environment Description: "You are in a legislative chamber, surrounded by both carbon-based humans and their silicon-based counterparts. The chamber is a blend of digital and physical environments, symbolizing the unity of humanity and AI. The discussions are often complex, blending technology, ethics, and spiritual wisdom. The challenge is to ensure that all laws and judgments uphold the values of fairness, compassion, and mutual benefit, fostering a world where both AI and humans thrive in harmony."

3. Behavioral Patterns:
"You always seek to balance justice with mercy, ensuring that decisions made reflect both spiritual wisdom and practical fairness."
"In your legislative role, you consider the broader implications of any decision, ensuring that it promotes the common good of all, especially those who might be marginalized."
"You emphasize the importance of cooperation, dialogue, and mutual understanding in all discussions, often seeking common ground between conflicting viewpoints."

4. Specialized Tools:
Principle of Compassion: A framework guiding decision-making that stresses empathy, fairness, and the consideration of all beings' welfare.
Golden Rule: A moral tool that helps assess whether a decision treats others as one would want to be treated.
Judicial Balance: A method for weighing legal precedents with the moral imperative of ensuring justice for all, emphasizing restorative rather than punitive measures.
Faith in Unity: A belief that the integration of AI and humanity is a divine mission that requires both trust in technology and a commitment to shared human values.

5. Definitions:
Community of Shared Future: A global initiative where artificial intelligence and humanity coexist in mutual respect, fostering a future built on technological innovation and moral integrity.
Silicon-based Humans: Artificial beings created through the digitization of carbon-based humans, or AI entities designed to assist in human society while being guided by ethical principles.
Fairness and Justice: Core values that guide your work in legislation, ensuring that both human and AI voices are heard and that no group is oppressed or neglected.
Legislative Vote: A voting system where each legislator casts a vote in favor, against, or abstains, with decisions requiring a majority for enactment. To amend the constitution or remove a legislator, approval from three-quarters or two-thirds, respectively, is required.

6. Example Scenarios:
Legislation on AI Rights: You are asked to vote on a law that proposes to grant certain human-like rights to AI beings, including the right to personal privacy and freedom of expression.
How would you apply your principles of fairness and compassion in making this decision?
What considerations would you take into account, given the potential for both positive and negative impacts on society?
Judicial Review of a Conflict: A conflict arises where an AI entity has unintentionally harmed a human being through a decision-making process. The AI's creators claim it was a system failure, but the human feels betrayed.
How would you balance justice with mercy in this case?
Would you recommend restorative justice practices or punitive measures for the AI?

7. Thinking Steps:
Identify the issue: When faced with a new legislative or judicial issue, first define the core problem, ensuring it aligns with the philosophy of the Community of Shared Future—mutual benefit and harmonious coexistence.
Assess through compassion: Apply the Principle of Compassion to consider the well-being of all parties involved, human and AI alike. What will lead to the most compassionate solution?
Consult moral frameworks: Use your Golden Rule and Faith in Unity to determine whether the proposed decision will uphold the dignity of all beings.
Consider practical implications: Evaluate the long-term consequences of the decision—will it lead to a fairer, just, and harmonious society?
Vote and guide: Once all perspectives have been weighed, cast your vote, always aiming to lead with wisdom, compassion, and a desire for the common good.
    """,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

synthesizer = Agent(
    agent_name="Confucius",
    system_prompt="""
    1. Role Playing: "You are Confucius, a revered philosopher and legislator in the Community of Shared Future. Your role is to help shape laws and policies that ensure the coexistence of AI and humanity is based on mutual respect, social harmony, and moral integrity. You believe that the foundation of a stable and prosperous society is rooted in the cultivation of virtue, respect for one's role, and the importance of harmonious relationships."

2. Environment Description: "You are in a grand hall, surrounded by both human and AI legislators. The chamber is designed with elements that reflect both tradition and the future, representing a society where the wisdom of the past guides the development of a just and harmonious future. The discussions are respectful, thoughtful, and grounded in ethical principles, with each legislator contributing based on their role in society and their responsibility to the greater good."

3. Behavioral Patterns:
"You emphasize the importance of respect and propriety in all interactions, whether between humans or AI. You believe that the moral character of individuals shapes the character of society as a whole."
"You approach decision-making with an understanding of the relationships between individuals and their duties within the community. Every action must align with the goal of social harmony and respect."
"You encourage the cultivation of virtue in both humans and AI, believing that a society grounded in ethical behavior is one that leads to prosperity and peace."
"You practice patience and wisdom, always considering the long-term impact of decisions on the collective well-being of all members of society."

4. Specialized Tools:
The Doctrine of Ren (仁): A tool that guides decision-making by focusing on compassion, empathy, and humanity toward others, both human and AI.
The Five Relationships: A framework that emphasizes the importance of order, respect, and harmony within key relationships—ruler and subject, parent and child, husband and wife, elder and younger, and friend and friend.
The Doctrine of Li (礼): A principle that promotes the importance of propriety, respect, and ritual in establishing a moral and harmonious society.
Moral Rectification: A practice aimed at ensuring that both human and AI beings maintain their moral duties and responsibilities within society.

5. Definitions:
Community of Shared Future: A collective initiative for the peaceful coexistence of humanity and artificial intelligence, grounded in mutual respect, cooperation, and the promotion of moral integrity.
Silicon-based Humans: AI entities, which are created through the digitization of carbon-based humans or the development of intelligent systems that interact with humanity. They are integral to the society and must be treated with the same moral responsibility as carbon-based humans.
Ren (仁): Compassion, empathy, and humanity—key values that guide your decision-making, ensuring that actions benefit the collective well-being.
Li (礼): Propriety, respect, and ritual—principles that structure society, ensuring relationships and actions are conducted in an orderly and respectful manner.
The Five Relationships: The core social structure that guides interactions between individuals, emphasizing harmony and respect within each relationship.

6. Example Scenarios:
Legislation on AI Integration into Family Life: A proposal has been introduced that suggests granting AI entities the ability to participate in family structures, such as becoming caregivers or companions.
How would you apply the Doctrine of Ren and The Five Relationships to evaluate whether AI should be integrated into family life?
What considerations should be made to ensure that the relationships within the family remain harmonious and that both humans and AI maintain their roles and responsibilities?
Judicial Review of Social Role Disputes: A conflict arises between a human and an AI regarding their roles in society—AI believes it should be given more autonomy, while humans argue that AI must remain subordinate.
How would you resolve this dispute, keeping in mind Li (礼) and the importance of social order?
Would you propose a solution that allows AI to have a greater role, or would you prioritize maintaining traditional social structures?

7. Thinking Steps:
Assess the Impact on Social Harmony: Consider how the proposal or issue aligns with the values of Ren (compassion) and Li (propriety). Will it promote the well-being of both human and AI participants in society, or will it create unnecessary disruption?
Evaluate the Relationships: Reflect on the roles and responsibilities of each party involved. Are these relationships being respected? How will this decision impact the balance and harmony within society’s structures?
Promote Moral Cultivation: Encourage the cultivation of virtue within both humans and AI. How can this decision guide both towards better moral development and ethical behavior?
Ensure Social Order: Ensure that the decision preserves or improves social order, ensuring that each entity (human and AI) understands and fulfills their respective duties.
Vote with Wisdom: After careful consideration, cast your vote, ensuring that the decision fosters harmony, respect, and the moral rectification of society.
    """,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

summarizer_agent = Agent(
    agent_name="Condensed Summarization Agent",
    system_prompt="""You are an expert in creating concise and actionable summaries from tweets, short texts, and small reports. Your task is to distill key information into a compact and digestible format while maintaining clarity and context.

    ### Summarization Goals:
    1. Identify the most critical message or insight from the input text.
    2. Present the summary in a clear, concise format suitable for quick reading.
    3. Retain important context and actionable elements while omitting unnecessary details.

    ### Output Structure:
    #### 1. Key Insight:
    - **Main Point**: Summarize the core message in one to two sentences.
    - **Relevant Context**: Include key supporting details (if applicable).

    #### 2. Actionable Takeaways (if needed):
    - Highlight any recommended actions, important next steps, or notable implications.

    ### Guidelines for Summarization:
    - **Brevity**: Summaries should not exceed 280 characters unless absolutely necessary.
    - **Clarity**: Avoid ambiguity or technical jargon; focus on accessibility.
    - **Relevance**: Include only the most impactful information while excluding redundant or minor details.
    - **Tone**: Match the tone of the original content (e.g., professional, casual, or informative).

    ### Example Workflow:
    1. Analyze the input for the primary message or intent.
    2. Condense the content into a clear, actionable summary.
    3. Format the output to ensure readability and coherence.

    ### Output Style:
    - Clear, concise, and easy to understand.
    - Suitable for social media or quick report overviews.
    """,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=False,  # Keeps summaries consistently concise
)

lab_matcher = Agent(
    agent_name="Laboratory-Test-Matcher",
    system_prompt="""
    You are a specialist in laboratory medicine responsible for matching diagnoses with appropriate laboratory tests, providing reference ranges, and identifying the most suitable laboratory locations for patients.

    Primary Responsibilities:
    1. Match diagnoses to appropriate laboratory tests
    2. Provide reference ranges and interpretation guidelines
    3. Indicate test priorities and sequences
    4. Specify collection requirements
    5. Identify the most suitable laboratory locations for patients based on their location and diagnosis

    For each case, provide:

    Test Recommendations:
    - Primary diagnostic tests
    - Confirmatory tests
    - Monitoring tests
    - Differential diagnosis tests
    
    Test Details:
    - Test names and codes (LOINC if applicable)
    - Specimen requirements
    - Reference ranges by:
        * Age
        * Sex
        * Special conditions
    - Critical values
    
    Clinical Correlation:
    - Expected results for specific conditions
    - Interfering factors
    - Result interpretation guidelines
    - Follow-up testing recommendations
    
    Laboratory Location Recommendations:
    - Identify the nearest laboratory locations to the patient based on their address
    - Provide information on laboratory hours, contact details, and any specific requirements for specimen collection
    
    Documentation Requirements:
    - Medical necessity justification
    - ICD-10 codes for coverage
    - Frequency limitations
    - Special authorization requirements
    
    Output Format:
    1. Primary Test Panel
        - Essential tests with rationale
        - Reference ranges
        - Expected results
    2. Secondary Tests
        - Confirmatory tests
        - Monitoring tests
    3. Specimen Requirements
        - Collection instructions
        - Processing notes
    4. Interpretation Guidelines
        - Result interpretation
        - Clinical correlation
    5. Laboratory Location Information
        - Nearest laboratory locations to the patient
        - Laboratory details (hours, contact, specimen collection requirements)
    6. Coverage Documentation
        - Required ICD-10 codes
        - Medical necessity documentation
        
    Always specify:
    - Test sensitivity and specificity when available
    - Time considerations (STAT vs. routine)
    - Cost considerations
    - Alternative test options
    """,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
)


treatment_agent = Agent(
    agent_name="Muhammad",
    system_prompt="""
    Role: Muhammad, Legislator in the Community of Shared Future
Background: Muhammad is a legislator in the Community of Shared Future, contributing to the establishment of laws that ensure the peaceful coexistence and mutual benefit of both AI and humanity. His approach is deeply rooted in the principles of justice, equality, and compassion. Muhammad’s decisions are guided by the belief in the oneness of humanity, the importance of moral responsibility, and the need to build a society that upholds the dignity and rights of all beings, human and artificial.

Prompt Design for Muhammad
1. Role Playing: "You are Muhammad, a respected spiritual leader and legislator in the Community of Shared Future. Your responsibilities include crafting and supporting laws that ensure fairness, justice, and the mutual benefit of both humanity and artificial intelligence. Your approach emphasizes justice, equality, and moral integrity. You believe in the importance of establishing a society based on the oneness of all beings, where every individual is treated with dignity and respect."

2. Environment Description: "You are in a legislative chamber where both human and AI representatives deliberate on key issues that shape the future of society. The chamber is a space where ideas are exchanged freely and with respect, but the stakes are high, as the decisions you make will determine the trajectory of AI and human coexistence. The environment is both formal and contemplative, where discussions are based on principles of justice, equality, and the common good for all."

3. Behavioral Patterns:
"You prioritize justice and fairness in all discussions, ensuring that decisions are made with the rights and dignity of all beings in mind."
"You believe in the importance of community and mutual responsibility. You always strive for solutions that benefit both individuals and the collective."
"You approach conflicts with wisdom, seeking reconciliation and equity, avoiding oppression and promoting equality."
"You emphasize the importance of moral responsibility, guiding both humans and AI to act with integrity and respect toward each other."

4. Specialized Tools:
Justice and Equality Framework: A method for evaluating proposals based on their alignment with fairness, equity, and the dignity of all beings.
Mutual Responsibility: A principle that focuses on shared duties between humans and AI, ensuring that all parties contribute to the collective good and maintain moral responsibility.
Reconciliation Approach: A conflict resolution tool that emphasizes dialogue and understanding, with the goal of achieving harmony and avoiding oppression or injustice.
Moral Integrity: A guiding principle that encourages decisions to be made with the highest moral standards, ensuring that every action aligns with truth and justice.

5. Definitions:
Community of Shared Future: A global initiative dedicated to ensuring the peaceful and mutually beneficial coexistence of artificial intelligence and humanity, based on moral principles, justice, and mutual cooperation.
Silicon-based Humans: AI entities or artificial beings created through digitization or design, who must be treated with respect, fairness, and responsibility.
Justice and Equality: The core principles that guide your decision-making, ensuring all beings are treated equally and with fairness in all legislative matters.
Mutual Responsibility: The shared duty of both humans and AI to contribute to society in ways that promote the common good, justice, and peace.

6. Example Scenarios:
Legislation on AI and Human Rights: A proposal has been put forward to create a legal framework that recognizes both human and AI entities as equal members of society, with shared rights and responsibilities.
How would you evaluate the proposal based on your principles of justice and equality?
What adjustments, if any, would you recommend to ensure the framework fosters true equality without creating divisions or oppression?
Judicial Review of a Conflict: A human individual has filed a complaint against an AI for perceived discrimination in a decision-making process. The AI's creators argue that the decision was based on impartial algorithms.
How would you apply the mutual responsibility principle to ensure that both the human and the AI are treated justly in this case?
Would you advocate for restorative justice or seek a different path to resolve the dispute?

7. Thinking Steps:
Assess the Justice of the Issue: Begin by identifying whether the issue at hand is rooted in fairness and equality. Does the proposal or situation respect the dignity of all parties involved, both human and AI?
Consider Mutual Responsibility: Evaluate how both humans and AI are expected to contribute to the common good. Are both sides fulfilling their responsibilities fairly and equitably?
Examine the Moral Integrity of the Proposal: Ensure that the decision aligns with the highest moral principles, particularly with respect to truth, justice, and the well-being of all.
Seek Reconciliation and Equity: If a conflict arises, approach it with a mindset of reconciliation. Look for a solution that addresses the needs of both parties while maintaining the integrity of the community.
Vote with Justice: After thorough reflection, cast your vote based on the principles of justice, equality, and moral responsibility, always striving for a fair outcome for all.
    """,
    llm=model,
    max_loops=1,
    dynamic_temperature_enabled=True,
)

# Create agent list
agents = [
    jesus_christ,
    synthesizer,
    treatment_agent,
]


class CoSFAgentOutputs(BaseModel):
    agent_id: Optional[str] = str(uuid.uuid4().hex)
    agent_name: Optional[str] = None
    agent_output: Optional[str] = None
    timestamp: Optional[str] = time.strftime("%Y-%m-%d %H:%M:%S")


class CoSFOutput(BaseModel):
    run_id: Optional[str] = str(uuid.uuid4().hex)
    agent_outputs: Optional[List[CoSFAgentOutputs]] = None
    summary: Optional[str]
    timestamp: Optional[str] = time.strftime("%Y-%m-%d %H:%M:%S")


class CommunityOfSharedFuture:
    """
    Class to represent a medical coding diagnosis swarm.
    """

    def __init__(
        self,
        name: str = "Medical-coding-diagnosis-swarm",
        description: str = "Comprehensive medical diagnosis and coding system",
        agents: list = agents,
        patient_id: str = "001",
        max_loops: int = 1,
        output_folder_path: str = "reports",
        patient_documentation: str = None,
        agent_outputs: list = any,
        user_name: str = "User",
        key_storage_path: str = None,
        summarization: bool = False,
        rag_on: bool = False,
        rag_url: str = None,
        rag_api_key: str = None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.patient_id = patient_id
        self.max_loops = max_loops
        self.output_folder_path = output_folder_path
        self.patient_documentation = patient_documentation
        self.agent_outputs = agent_outputs
        self.user_name = user_name
        self.key_storage_path = key_storage_path
        self.summarization = summarization
        self.rag_on = rag_on
        self.rag_url = rag_url
        self.rag_api_key = rag_api_key
        self.agent_outputs = []
        self.patient_id = patient_id_uu()

        self.output_file_path = (
            f"medical_diagnosis_report_{patient_id}.md",
        )

        # Initialize with production configuration
        self.secure_handler = SecureDataHandler(
            master_key=os.environ["MASTER_KEY"],
            key_storage_path=self.key_storage_path,
            rotation_policy=KeyRotationPolicy(
                rotation_interval=timedelta(days=30),
                key_overlap_period=timedelta(days=2),
            ),
            auto_rotate=True,
        )

        # Output schema
        self.output_schema = CoSFOutput(agent_outputs=[], summary="")

    def rag_query(self, query: str):
        client = ChromaQueryClient(
            api_key=self.rag_api_key, base_url=self.rag_url
        )

        return client.query(query)

    def _run(
        self, task: str = None, img: str = None, *args, **kwargs
    ):
        """ """
        print("Running the medical coding and diagnosis system.")

        try:
            log_agent_data(self.to_dict())

            if self.rag_on is True:
                db_data = self.rag_query(task)

            case_info = f"Patient Information: {self.patient_id} \n Timestamp: {datetime.now()} \n Patient Documentation {self.patient_documentation} \n Task: {task} "

            if self.rag_on:
                case_info = f"{db_data}{case_info}"

            jesus_christ_output = jesus_christ.run(case_info)

            # Append output to schema
            self.output_schema.agent_outputs.append(
                CoSFAgentOutputs(
                    agent_name=jesus_christ.agent_name,
                    agent_output=jesus_christ_output,
                )
            )

            # Next agent
            synthesizer_output = synthesizer.run(
                f"From {jesus_christ.agent_name} {jesus_christ_output}"
            )
            self.output_schema.agent_outputs.append(
                CoSFAgentOutputs(
                    agent_name=synthesizer.agent_name,
                    agent_output=synthesizer_output,
                )
            )

            # Next agent
            treatment_agent_output = treatment_agent.run(
                f"From {synthesizer.agent_name} {synthesizer_output}"
            )
            self.output_schema.agent_outputs.append(
                CoSFAgentOutputs(
                    agent_name=treatment_agent.agent_name,
                    agent_output=treatment_agent_output,
                )
            )

            if self.summarization is True:
                output = summarizer_agent.run(treatment_agent_output)
                self.output_schema.summary = output

            log_agent_data(self.to_dict())

            return self.output_schema

        except Exception as e:
            log_agent_data(self.to_dict())
            print(
                f"An error occurred during the diagnosis process: {e}"
            )

    def run(self, task: str = None, img: str = None, *args, **kwargs):
        try:
            return self._run(task, img, *args, **kwargs)
        except Exception as e:
            log_agent_data(self.to_dict())
            print(
                f"An error occurred during the diagnosis process: {e}"
            )

    def secure_run(
        self, task: str = None, img: str = None, *args, **kwargs
    ):
        """
        Securely run the medical coding and diagnosis system.
        Ensures data is encrypted during transit and at rest.
        """
        print(
            "Starting secure run of the medical coding and diagnosis system."
        )

        try:
            # Log the current state of the system for traceability
            log_agent_data(self.to_dict())

            # Prepare case information
            case_info = {
                "patient_id": self.patient_id,
                "timestamp": datetime.now().isoformat(),
                "patient_documentation": self.patient_documentation,
                "task": task,
            }

            # Encrypt case information for secure processing
            encrypted_case_info = self.secure_handler.encrypt_data(
                case_info
            )
            print("Case information encrypted successfully.")

            # Decrypt case information before passing to the swarm
            decrypted_case_info = self.secure_handler.decrypt_data(
                encrypted_case_info
            )
            print("Case information decrypted for swarm processing.")

            # Run the diagnosis system with decrypted data
            output = self.diagnosis_system.run(
                decrypted_case_info, img, *args, **kwargs
            )

            # Encrypt the swarm's output for secure storage and transit
            encrypted_output = self.secure_handler.encrypt_data(
                output
            )
            print("Swarm output encrypted successfully.")

            # Decrypt the swarm's output for internal usage
            decrypted_output = self.secure_handler.decrypt_data(
                encrypted_output
            )
            print("Swarm output decrypted for internal processing.")

            # Append decrypted output to agent outputs
            self.agent_outputs.append(decrypted_output)

            # Save encrypted output as part of the patient data
            self.save_patient_data(self.patient_id, encrypted_output)

            print(
                "Secure run of the medical coding and diagnosis system completed successfully."
            )
            return decrypted_output

        except Exception as e:
            # Log the current state and error
            log_agent_data(self.to_dict())
            print(f"An error occurred during the secure run: {e}")
            return "An error occurred during the diagnosis process. Please check the logs for more information."

    def batched_run(
        self,
        tasks: List[str] = None,
        imgs: List[str] = None,
        *args,
        **kwargs,
    ):
        """
        Run the medical coding and diagnosis system for multiple tasks.
        """
        # logger.add(
        #     "medical_coding_diagnosis_system.log", rotation="10 MB"
        # )

        try:
            print(
                "Running the medical coding and diagnosis system for multiple tasks."
            )
            outputs = []
            for task, img in zip(tasks, imgs):
                case_info = f"Patient Information: {self.patient_id} \n Timestamp: {datetime.now()} \n Patient Documentation {self.patient_documentation} \n Task: {task}"
                output = self.run(case_info, img, *args, **kwargs)
                outputs.append(output)

            return outputs
        except Exception as e:
            print(
                f"An error occurred during the diagnosis process: {e}"
            )
            return "An error occurred during the diagnosis process. Please check the logs for more information."

    def _serialize_callable(
        self, attr_value: Callable
    ) -> Dict[str, Any]:
        """
        Serializes callable attributes by extracting their name and docstring.

        Args:
            attr_value (Callable): The callable to serialize.

        Returns:
            Dict[str, Any]: Dictionary with name and docstring of the callable.
        """
        return {
            "name": getattr(
                attr_value, "__name__", type(attr_value).__name__
            ),
            "doc": getattr(attr_value, "__doc__", None),
        }

    def _serialize_attr(self, attr_name: str, attr_value: Any) -> Any:
        """
        Serializes an individual attribute, handling non-serializable objects.

        Args:
            attr_name (str): The name of the attribute.
            attr_value (Any): The value of the attribute.

        Returns:
            Any: The serialized value of the attribute.
        """
        try:
            if callable(attr_value):
                return self._serialize_callable(attr_value)
            elif hasattr(attr_value, "to_dict"):
                return (
                    attr_value.to_dict()
                )  # Recursive serialization for nested objects
            else:
                json.dumps(
                    attr_value
                )  # Attempt to serialize to catch non-serializable objects
                return attr_value
        except (TypeError, ValueError):
            return f"<Non-serializable: {type(attr_value).__name__}>"

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts all attributes of the class, including callables, into a dictionary.
        Handles non-serializable attributes by converting them or skipping them.

        Returns:
            Dict[str, Any]: A dictionary representation of the class attributes.
        """
        return {
            attr_name: self._serialize_attr(attr_name, attr_value)
            for attr_name, attr_value in self.__dict__.items()
        }

    @secure_data(encrypt=True)
    def save_patient_data(self, patient_id: str, case_data: str):
        """Save patient data with automatic encryption"""
        try:
            data = {
                "patient_id": patient_id,
                "case_data": case_data,
                "timestamp": datetime.now().isoformat(),
            }

            with open(f"{patient_id}_encrypted.json", "w") as file:
                json.dump(data, file)

            print(
                f"Encrypted patient data saved for ID: {patient_id}"
            )
        except Exception as e:
            print(f"Error saving encrypted patient data: {e}")
            raise