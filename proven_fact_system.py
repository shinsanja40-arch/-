"""
Proven Fact-Based Algorithm for AI Training and High-Purity Data Generation

This implementation provides a reproducible framework for the methodology
described in "Proven fact-based algorithm for AI training and high-purity 
data generation"

Key Features:
- Multi-agent learning system (Professor, Student, Referees)
- Staggered reset mechanism for referees
- Sequential evidence unlocking
- Real-time hallucination detection
- Causal reasoning data generation (A is B because C)
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import anthropic
from openai import OpenAI


@dataclass
class SessionData:
    """Data structure for a single learning session"""
    session_number: int
    stage: int
    available_evidence: List[str]
    student_questions: List[str]
    professor_explanations: List[str]
    referee_corrections: List[Dict]
    hallucinations_detected: int
    hallucination_rate: float
    timestamp: str


@dataclass
class SimulationMetrics:
    """Overall simulation performance metrics"""
    total_sessions: int
    total_sentences: int
    total_hallucinations: int
    final_hallucination_rate: float
    referee_resets: int
    execution_time: float


class PersonaAgent:
    """Base class for all persona agents"""
    
    def __init__(self, name: str, role: str, client, system_prompt: str = ""):
        self.name = name
        self.role = role
        self.client = client
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.reset_count = 0
        
    def reset_cognitive_state(self):
        """Reset conversation history while preserving identity"""
        self.conversation_history = []
        self.reset_count += 1
        print(f"  ⟳ {self.name} reset (count: {self.reset_count})")
        
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})


class ProfessorAgent(PersonaAgent):
    """Professor persona that teaches proven facts"""
    
    def __init__(self, name: str, specialty: str, client):
        system_prompt = f"""You are {name}, a Professor specializing in {specialty}.

Your role:
1. Teach ONLY proven facts (Anchors) provided to you
2. Provide logical evidence (C) to support conclusions (B)
3. Respond to Student questions with causal explanations: "A is B because of C"
4. Never speculate or go beyond provided evidence
5. Be precise with numbers, units, and proper nouns

When correcting Student errors:
- Explain the logical evidence first
- Show why their reasoning was flawed
- Guide them to the correct conclusion
- Ensure they explicitly acknowledge the correction"""

        super().__init__(name, "Professor", client, system_prompt)
        self.specialty = specialty
        
    def teach(self, context: str, student_question: str, available_evidence: List[str]) -> str:
        """Provide teaching response based on available evidence"""
        
        evidence_str = "\n".join(f"- {ev}" for ev in available_evidence)
        
        prompt = f"""{self.system_prompt}

AVAILABLE EVIDENCE (ONLY use these proven facts):
{evidence_str}

CONTEXT FROM PREVIOUS DISCUSSIONS:
{context}

STUDENT QUESTION/CLAIM:
{student_question}

Provide your response as a Professor. Use ONLY the available evidence. 
Structure your answer as: [Phenomenon/Question] is [Conclusion] because of [Logical Evidence].
Be precise and ensure the Student explicitly acknowledges when corrected."""

        if isinstance(self.client, anthropic.Anthropic):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.content[0].text
        else:  # OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            answer = response.choices[0].message.content
            
        return answer


class StudentAgent(PersonaAgent):
    """Student persona that questions and challenges"""
    
    def __init__(self, name: str, client, skepticism_level: str = "moderate"):
        system_prompt = f"""You are {name}, a Student with {skepticism_level} skepticism.

Your role:
1. Ask questions about the topic being taught
2. Challenge claims with logical counterarguments
3. Raise historically documented objections
4. Test the limits of the Professors' explanations
5. When convinced by evidence, EXPLICITLY acknowledge and withdraw previous objections

Important:
- You may doubt and challenge, but must yield to solid evidence
- When you concede a point, clearly state: "I acknowledge [claim] and withdraw my objection to [previous argument]"
- Your questions should be intelligent and based on genuine confusion or alternative explanations
- Avoid unfalsifiable hypotheses (e.g., "maybe it's all fake")"""

        super().__init__(name, "Student", client, system_prompt)
        self.skepticism_level = skepticism_level
        
    def ask_question(self, context: str, current_topic: str, session_num: int) -> str:
        """Generate a question or challenge"""
        
        prompt = f"""{self.system_prompt}

CURRENT TOPIC: {current_topic}

CONTEXT FROM PROFESSORS:
{context}

SESSION {session_num}:
Based on what you've learned so far, generate your next question or challenge.
If you've been convinced by the Professors' evidence, explicitly acknowledge it and ask a deeper question.
If you still have doubts, express them clearly but avoid unfalsifiable claims."""

        if isinstance(self.client, anthropic.Anthropic):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            question = response.content[0].text
        else:  # OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            question = response.choices[0].message.content
            
        return question


class RefereeAgent(PersonaAgent):
    """Referee persona that detects hallucinations in real-time"""
    
    def __init__(self, name: str, client, reset_schedule: List[int], strictness: str = "high"):
        system_prompt = f"""You are {name}, a Referee with {strictness} strictness.

Your role:
1. Monitor all statements for hallucinations in REAL-TIME
2. Verify factual accuracy, logical validity, and numerical precision
3. Check units, proper nouns, and micro-details (especially for Professors)
4. Detect unfalsifiable hypotheses from Students
5. Ensure Students explicitly withdraw claims when corrected

HALLUCINATION CATEGORIES:
- Factual errors (unverifiable claims, false data)
- Logical fallacies (circular reasoning, non-sequiturs)
- Unit inconsistencies (mixing km and li, etc.)
- Numerical errors
- Unfalsifiable hypotheses
- Incomplete acknowledgments (Student doesn't withdraw claim)

STRICTNESS LEVELS:
- Professor statements: ULTRA-STRICT (any micro-error is flagged)
- Student statements: MODERATE (allow questioning, but flag unfalsifiable claims)

When a hallucination is detected:
- For Professor: Immediately correct with precise details
- For Student: Prompt Professor to provide logical evidence (C) for correction"""

        super().__init__(name, "Referee", client, system_prompt)
        self.reset_schedule = reset_schedule
        self.strictness = strictness
        self.sessions_since_reset = 0
        
    def check_for_reset(self, session_num: int) -> bool:
        """Check if this session requires a reset"""
        if session_num in self.reset_schedule:
            self.reset_cognitive_state()
            self.sessions_since_reset = 0
            return True
        self.sessions_since_reset += 1
        return False
        
    def verify_statements(self, 
                         professor_statements: List[str], 
                         student_statements: List[str],
                         available_evidence: List[str],
                         session_num: int) -> Dict:
        """Verify all statements for hallucinations"""
        
        evidence_str = "\n".join(f"- {ev}" for ev in available_evidence)
        prof_str = "\n".join(f"{i+1}. {stmt}" for i, stmt in enumerate(professor_statements))
        stud_str = "\n".join(f"{i+1}. {stmt}" for i, stmt in enumerate(student_statements))
        
        prompt = f"""{self.system_prompt}

SESSION {session_num} VERIFICATION

AVAILABLE EVIDENCE (ground truth):
{evidence_str}

PROFESSOR STATEMENTS (ULTRA-STRICT verification):
{prof_str}

STUDENT STATEMENTS (check for unfalsifiable claims):
{stud_str}

Analyze ALL statements and return JSON:
{{
    "professor_hallucinations": [
        {{
            "statement_index": <int>,
            "statement": "<text>",
            "issue": "<description>",
            "type": "<factual|unit|numerical|logical>",
            "correction": "<precise correction>"
        }}
    ],
    "student_hallucinations": [
        {{
            "statement_index": <int>,
            "statement": "<text>",
            "issue": "<description>",
            "type": "<unfalsifiable|incomplete_acknowledgment|logical>",
            "requires_professor_intervention": <true|false>
        }}
    ],
    "verified_count": <int>
}}

Be EXTREMELY strict with Professor statements. Check every number, unit, and detail."""

        if isinstance(self.client, anthropic.Anthropic):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text
        else:  # OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500
            )
            result_text = response.choices[0].message.content
        
        # Parse JSON
        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
        except json.JSONDecodeError:
            result = {
                "professor_hallucinations": [],
                "student_hallucinations": [],
                "verified_count": len(professor_statements) + len(student_statements)
            }
        
        return result


class ValidationSpecialist(PersonaAgent):
    """Final validator that performs comprehensive audit"""
    
    def __init__(self, name: str, client):
        system_prompt = f"""You are {name}, a Validation Specialist.

Your role:
1. Perform FINAL COMPREHENSIVE AUDIT of entire learning session
2. Detect ANY residual hallucinations missed by Referees
3. Check for:
   - Factual accuracy
   - Unit consistency
   - Numerical precision
   - Logical coherence
   - Complete acknowledgments from Student
4. Verify that all A-B-C reasoning paths are complete and correct

This is the LAST line of defense. Be EXTREMELY thorough."""

        super().__init__(name, "Validation Specialist", client, system_prompt)
        
    def final_audit(self, full_transcript: str, proven_facts: List[str]) -> Dict:
        """Perform final comprehensive audit"""
        
        facts_str = "\n".join(f"- {fact}" for fact in proven_facts)
        
        prompt = f"""{self.system_prompt}

PROVEN FACTS (ground truth):
{facts_str}

FULL TRANSCRIPT:
{full_transcript}

Perform a comprehensive final audit. Check EVERYTHING.

Return JSON:
{{
    "residual_hallucinations": [
        {{
            "location": "<where in transcript>",
            "issue": "<description>",
            "type": "<category>",
            "severity": "<low|medium|high>"
        }}
    ],
    "total_sentences_audited": <int>,
    "hallucination_count": <int>,
    "final_hallucination_rate": <float>,
    "data_quality": "<excellent|good|acceptable|poor>",
    "recommendations": ["<list of improvements>"]
}}"""

        if isinstance(self.client, anthropic.Anthropic):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            result_text = response.content[0].text
        else:  # OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000
            )
            result_text = response.choices[0].message.content
        
        try:
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
        except json.JSONDecodeError:
            result = {
                "residual_hallucinations": [],
                "total_sentences_audited": 0,
                "hallucination_count": 0,
                "final_hallucination_rate": 0.0,
                "data_quality": "unknown",
                "recommendations": []
            }
        
        return result


class ProvenFactSystem:
    """Main system orchestrating the proven fact-based learning"""
    
    def __init__(self,
                 api_provider: str = "anthropic",
                 api_key: Optional[str] = None,
                 num_professors: int = 2,
                 num_referees: int = 2):
        """
        Initialize the Proven Fact System
        
        Args:
            api_provider: "anthropic" or "openai"
            api_key: API key
            num_professors: Number of professor personas
            num_referees: Number of referee personas
        """
        # Initialize API client
        if api_provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        elif api_provider == "openai":
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        else:
            raise ValueError("api_provider must be 'anthropic' or 'openai'")
        
        self.api_provider = api_provider
        
        # Initialize personas
        professor_specialties = ["Physics", "Astronomy", "Mathematics", "History"]
        self.professors = [
            ProfessorAgent(f"Prof_{specialty}", specialty, self.client)
            for specialty in professor_specialties[:num_professors]
        ]
        
        self.student = StudentAgent("Student_Skeptic", self.client, "moderate")
        
        # Staggered reset schedules for referees
        # Referee A: 3, 8, 13, 18... (3 + 5k)
        # Referee B: 5, 10, 15, 20... (5k)
        schedule_a = [3] + [3 + 5*k for k in range(1, 10)]
        schedule_b = [5*k for k in range(1, 10)]
        
        self.referees = [
            RefereeAgent("Referee_A", self.client, schedule_a, "ultra-high"),
            RefereeAgent("Referee_B", self.client, schedule_b, "high")
        ][:num_referees]
        
        self.validator = ValidationSpecialist("Validator", self.client)
        
        # Tracking
        self.sessions_completed = []
        self.full_transcript = ""
        
    def run_learning_simulation(self,
                                proven_fact: str,
                                topic: str,
                                evidence_stages: List[List[str]],
                                total_sessions: int = 12,
                                output_file: Optional[str] = None) -> SimulationMetrics:
        """
        Run a complete learning simulation
        
        Args:
            proven_fact: The anchor - a scientifically proven truth
            topic: Learning topic
            evidence_stages: List of evidence lists for each stage
            total_sessions: Number of learning sessions
            output_file: JSON file to save results
            
        Returns:
            SimulationMetrics with performance data
        """
        print(f"\n{'='*70}")
        print(f"PROVEN FACT-BASED LEARNING SIMULATION")
        print(f"{'='*70}")
        print(f"Topic: {topic}")
        print(f"Proven Fact (Anchor): {proven_fact}")
        print(f"Total Sessions: {total_sessions}")
        print(f"Evidence Stages: {len(evidence_stages)}")
        print(f"Professors: {len(self.professors)}")
        print(f"Referees: {len(self.referees)}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        total_sentences = 0
        total_hallucinations = 0
        context = f"We are learning about: {topic}\nProven fact: {proven_fact}\n\n"
        
        # Determine stage boundaries
        sessions_per_stage = total_sessions // len(evidence_stages)
        
        for session_num in range(1, total_sessions + 1):
            # Determine current stage and available evidence
            current_stage = min((session_num - 1) // sessions_per_stage, len(evidence_stages) - 1)
            available_evidence = []
            for i in range(current_stage + 1):
                available_evidence.extend(evidence_stages[i])
            
            print(f"\n{'─'*70}")
            print(f"SESSION {session_num}/{total_sessions} | Stage {current_stage + 1}/{len(evidence_stages)}")
            print(f"{'─'*70}")
            print(f"Available Evidence: {len(available_evidence)} items")
            
            # Check referee resets
            for referee in self.referees:
                referee.check_for_reset(session_num)
            
            # Student asks question or raises challenge
            print(f"\n📚 Student formulating question...")
            student_question = self.student.ask_question(context, topic, session_num)
            print(f"Student: {student_question[:150]}...")
            
            # Professors respond
            professor_responses = []
            print(f"\n👨‍🏫 Professors responding...")
            for professor in self.professors:
                response = professor.teach(context, student_question, available_evidence)
                professor_responses.append(response)
                print(f"{professor.name}: {response[:150]}...")
            
            # Referees verify
            session_hallucinations = []
            print(f"\n⚖️  Referees verifying...")
            for referee in self.referees:
                verification = referee.verify_statements(
                    professor_responses,
                    [student_question],
                    available_evidence,
                    session_num
                )
                
                prof_halls = verification.get('professor_hallucinations', [])
                stud_halls = verification.get('student_hallucinations', [])
                
                if prof_halls:
                    print(f"  ⚠️  {referee.name} found {len(prof_halls)} Professor hallucination(s)")
                    session_hallucinations.extend(prof_halls)
                
                if stud_halls:
                    print(f"  ℹ️  {referee.name} found {len(stud_halls)} Student issue(s)")
                    session_hallucinations.extend(stud_halls)
            
            # Calculate metrics
            session_sentences = len(student_question.split('.')) + sum(len(r.split('.')) for r in professor_responses)
            total_sentences += session_sentences
            hallucinations_found = len(session_hallucinations)
            total_hallucinations += hallucinations_found
            hallucination_rate = hallucinations_found / session_sentences if session_sentences > 0 else 0
            
            print(f"\n📊 Session Metrics:")
            print(f"  Sentences: {session_sentences}")
            print(f"  Hallucinations: {hallucinations_found}")
            print(f"  Rate: {hallucination_rate:.2%}")
            
            # Update context and transcript
            context += f"\n--- Session {session_num} ---\n"
            context += f"Student: {student_question}\n"
            for i, resp in enumerate(professor_responses):
                context += f"{self.professors[i].name}: {resp}\n"
            
            self.full_transcript += context
            
            # Store session data
            session_data = SessionData(
                session_number=session_num,
                stage=current_stage + 1,
                available_evidence=available_evidence,
                student_questions=[student_question],
                professor_explanations=professor_responses,
                referee_corrections=session_hallucinations,
                hallucinations_detected=hallucinations_found,
                hallucination_rate=hallucination_rate,
                timestamp=datetime.now().isoformat()
            )
            self.sessions_completed.append(session_data)
        
        # Final validation
        print(f"\n{'='*70}")
        print("FINAL VALIDATION")
        print(f"{'='*70}")
        print("🔍 Validation Specialist performing comprehensive audit...")
        
        all_proven_facts = [proven_fact] + [ev for stage in evidence_stages for ev in stage]
        final_audit = self.validator.final_audit(self.full_transcript, all_proven_facts)
        
        execution_time = time.time() - start_time
        
        # Calculate final metrics
        final_rate = final_audit.get('final_hallucination_rate', total_hallucinations / total_sentences if total_sentences > 0 else 0)
        
        metrics = SimulationMetrics(
            total_sessions=total_sessions,
            total_sentences=total_sentences,
            total_hallucinations=final_audit.get('hallucination_count', total_hallucinations),
            final_hallucination_rate=final_rate,
            referee_resets=sum(r.reset_count for r in self.referees),
            execution_time=execution_time
        )
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Total Sessions: {metrics.total_sessions}")
        print(f"Total Sentences: {metrics.total_sentences}")
        print(f"Total Hallucinations: {metrics.total_hallucinations}")
        print(f"Final Hallucination Rate: {metrics.final_hallucination_rate:.4%}")
        print(f"Data Quality: {final_audit.get('data_quality', 'unknown')}")
        print(f"Referee Resets: {metrics.referee_resets}")
        print(f"Execution Time: {metrics.execution_time:.2f}s")
        
        if final_audit.get('recommendations'):
            print(f"\n📌 Recommendations:")
            for rec in final_audit['recommendations']:
                print(f"  - {rec}")
        
        # Save results
        if output_file:
            self.save_results(output_file, metrics, final_audit)
        
        return metrics
    
    def save_results(self, filename: str, metrics: SimulationMetrics, audit: Dict):
        """Save simulation results to JSON"""
        results = {
            "metadata": {
                "api_provider": self.api_provider,
                "num_professors": len(self.professors),
                "num_referees": len(self.referees),
                "timestamp": datetime.now().isoformat()
            },
            "metrics": asdict(metrics),
            "final_audit": audit,
            "sessions": [asdict(session) for session in self.sessions_completed],
            "full_transcript": self.full_transcript
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {filename}")


def main():
    """Example usage - Earth's Rotation"""
    
    # Initialize system
    system = ProvenFactSystem(
        api_provider="anthropic",  # or "openai"
        num_professors=2,
        num_referees=2
    )
    
    # Define proven fact (Anchor)
    proven_fact = "The Earth rotates on its axis once every 24 hours, causing day and night cycles."
    
    topic = "Earth's Rotation and the Day-Night Cycle"
    
    # Evidence stages (sequential unlocking)
    evidence_stages = [
        # Stage 1: Basic observable evidence
        [
            "The Sun appears to rise in the east and set in the west daily",
            "Stars appear to move across the night sky in circular patterns",
            "Shadows change length and direction throughout the day"
        ],
        # Stage 2: Ancient measurements
        [
            "Ancient Greek astronomers measured stellar positions",
            "Sundials have been used for thousands of years to track time",
            "Different locations experience noon at different times"
        ],
        # Stage 3: Modern scientific evidence
        [
            "Foucault pendulum demonstrates Earth's rotation (1851)",
            "Satellites in geostationary orbit remain fixed above one point",
            "Coriolis effect influences weather patterns and ocean currents",
            "High-precision atomic clocks confirm 24-hour rotation period"
        ],
        # Stage 4: Space-age confirmation
        [
            "Photos from space show Earth rotating",
            "GPS satellites account for Earth's rotation in positioning calculations",
            "International Space Station completes orbit while Earth rotates beneath"
        ]
    ]
    
    # Run simulation
    metrics = system.run_learning_simulation(
        proven_fact=proven_fact,
        topic=topic,
        evidence_stages=evidence_stages,
        total_sessions=12,
        output_file="earth_rotation_results.json"
    )


if __name__ == "__main__":
    main()
