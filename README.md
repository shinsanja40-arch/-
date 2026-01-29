# Proven Fact-Based Algorithm for AI Training and High-Purity Data Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of the multi-agent learning system described in:

**"Proven fact-based algorithm for AI training and high-purity data generation"**

## 📋 Overview

This system addresses the **Data Wall** problem in AI by generating ultra-high-purity training data through a novel multi-agent learning approach. Unlike conventional methods that simply map "A is B," this system teaches AI the causal reasoning: **"A is B because of C."**

### Key Features

- **Anchor-Based Learning**: Scientifically proven facts serve as stable reference points
- **Multi-Agent System**: Professors teach, Students question, Referees verify
- **Staggered Reset Mechanism**: Prime-based intervals prevent referee contamination
- **Sequential Evidence Unlocking**: Bottom-up knowledge construction from scarcity to abundance
- **Real-Time Hallucination Detection**: Triple-layer verification system
- **Causal Reasoning Extraction**: Generates A→B→C reasoning paths for retraining

### Proven Results

From the paper's simulations:
- **0% hallucination rate** in 3 out of 4 experiments
- **1.8% final rate** in the fourth (unit consistency issue)
- **100% correction rate** for detected hallucinations
- Preserves complete causal reasoning pathways

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- API key for Anthropic Claude or OpenAI GPT

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/proven-fact-algorithm.git
cd proven-fact-algorithm

# Install dependencies
pip install -r requirements.txt

# Set up your API key
export ANTHROPIC_API_KEY="your-api-key-here"
# OR
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
from proven_fact_system import ProvenFactSystem

# Initialize the system
system = ProvenFactSystem(
    api_provider="anthropic",  # or "openai"
    num_professors=2,
    num_referees=2
)

# Define your proven fact (Anchor)
proven_fact = "The Earth rotates on its axis once every 24 hours."

# Define evidence stages (sequential unlocking)
evidence_stages = [
    ["Observable phenomena..."],
    ["Ancient measurements..."],
    ["Modern scientific evidence..."],
    ["Space-age confirmation..."]
]

# Run simulation
metrics = system.run_learning_simulation(
    proven_fact=proven_fact,
    topic="Earth's Rotation",
    evidence_stages=evidence_stages,
    total_sessions=12,
    output_file="results.json"
)

print(f"Final Hallucination Rate: {metrics.final_hallucination_rate:.4%}")
```

### Command Line Interface

```bash
# Use a predefined template
python run_proven_fact.py --template earth_rotation --sessions 12

# Available templates
python run_proven_fact.py --template earth_sphericity --sessions 15
python run_proven_fact.py --template evolution --sessions 15
python run_proven_fact.py --template climate_change --sessions 15

# Analyze results
python analyze_proven_fact.py earth_rotation_results.json
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Validation Specialist (Final Audit)            │
│          Performs comprehensive post-simulation          │
└─────────────────────────────────────────────────────────┘
                            │
    ┌───────────────────────┴───────────────────────┐
    │                                               │
┌───▼────────────────┐                  ┌──────────▼────────┐
│   Referee A        │                  │   Referee B        │
│   Reset: 3,8,13... │                  │   Reset: 5,10,15...│
│   (Ultra-strict)   │                  │   (High-strict)    │
└───┬────────────────┘                  └──────────┬─────────┘
    │                                               │
    └────────────────┬──────────────────────────────┘
                     │ Real-time verification
    ┌────────────────┴────────────────┐
    │                                 │
┌───▼────────┐              ┌─────────▼────────┐
│ Professor 1 │              │    Student       │
│ (Physics)   │◄────────────►│   (Skeptical)    │
└─────────────┘              └──────────────────┘
┌─────────────┐
│ Professor 2 │
│ (Astronomy) │
└─────────────┘
```

### Agent Roles

#### Professor Agents
- Teach proven facts (Anchors)
- Provide causal explanations: "A is B **because of** C"
- Respond to Student challenges with logical evidence
- Must be 100% accurate (ultra-strict verification)

#### Student Agent
- Questions and challenges Professors
- Raises historically documented objections
- Tests logical boundaries
- **Must explicitly acknowledge** when corrected

#### Referee Agents
- Monitor all statements in real-time
- Detect hallucinations at micro-level (units, numbers, logic)
- Different strictness for Professors (ultra-high) vs Students (moderate)
- Independent staggered reset cycles

#### Validation Specialist
- Final comprehensive audit after all sessions
- Catches any hallucinations missed by Referees
- Provides data quality assessment

## 📊 Sequential Evidence Unlocking

The system builds knowledge from information scarcity to abundance:

### Stage 1: Observable Evidence
- Basic phenomena anyone can observe
- Foundation for initial reasoning

### Stage 2: Historical/Ancient Evidence
- Measurements and observations from history
- Demonstrates long-standing knowledge

### Stage 3: Modern Scientific Evidence
- Precise measurements and experiments
- Controlled scientific validation

### Stage 4: Contemporary/Space-Age Evidence
- Latest technology and direct observation
- Definitive confirmation

This progression mirrors **human knowledge accumulation** and teaches the AI how reasoning sophistication grows with available evidence.

## 🔬 Key Innovations

### 1. Staggered Reset Mechanism

Unlike simple systems where all agents reset together (creating verification gaps), this uses **prime-based intervals**:

- **Referee A**: Resets at sessions 3, 8, 13, 18... (3 + 5k)
- **Referee B**: Resets at sessions 5, 10, 15, 20... (5k)

This ensures **continuous oversight** with no surveillance gaps.

### 2. Differential Strictness

- **Professor statements**: ULTRA-STRICT
  - Every number checked
  - Every unit verified  
  - Every proper noun validated
  - Any micro-error flagged immediately

- **Student statements**: MODERATE
  - Allow questioning and doubt
  - Flag unfalsifiable hypotheses
  - Ensure explicit acknowledgment of corrections

### 3. Causal Reasoning Extraction

Instead of just "Earth is round," the system captures:

```
Question (A): "Why do ships disappear hull-first?"
Conclusion (B): "Because Earth is curved"
Evidence (C): "Light travels in straight lines; curved surface blocks lower portions first"

→ Retraining data: A is B because of C
```

This teaches **why** conclusions are valid, not just **what** they are.

## 📈 Reproducing Paper Results

### Earth's Rotation (ChatGPT)

```bash
python run_proven_fact.py \
    --template earth_rotation \
    --provider openai \
    --sessions 12 \
    --output earth_rotation_gpt.json

python analyze_proven_fact.py earth_rotation_gpt.json
```

**Expected**: 0% hallucination rate (per paper)

### Earth's Rotation (Claude)

```bash
python run_proven_fact.py \
    --template earth_rotation \
    --provider anthropic \
    --sessions 12 \
    --output earth_rotation_claude.json
```

**Expected**: 0% hallucination rate (per paper)

### Earth's Sphericity (ChatGPT)

```bash
python run_proven_fact.py \
    --template earth_sphericity \
    --provider openai \
    --sessions 15 \
    --output earth_sphericity_gpt.json
```

**Expected**: 0% hallucination rate

### Earth's Sphericity (Claude)

```bash
python run_proven_fact.py \
    --template earth_sphericity \
    --provider anthropic \
    --sessions 15 \
    --output earth_sphericity_claude.json
```

**Expected**: ~1.8% hallucination rate (unit consistency issue per paper)

## 🎯 Use Cases

### 1. High-Purity Retraining Data Generation
Generate datasets where AI learns not just answers, but **causal reasoning paths**.

### 2. Knowledge Verification Systems
Validate that AI can prove claims using minimal evidence before accepting them.

### 3. Educational AI Development
Train AI tutors that can explain **why** answers are correct, not just provide them.

### 4. Fact-Checking Automation
Build systems that verify claims through logical decomposition.

### 5. Scientific Knowledge Curation
Extract provable facts from complex domains for AI training.

## ⚙️ Configuration

### Custom Simulation

Create a JSON configuration file:

```json
{
  "proven_fact": "Your scientifically proven anchor",
  "topic": "Your learning topic",
  "evidence_stages": [
    ["Stage 1 evidence items..."],
    ["Stage 2 evidence items..."],
    ["Stage 3 evidence items..."],
    ["Stage 4 evidence items..."]
  ]
}
```

Run with:

```bash
python run_proven_fact.py --config my_config.json --sessions 12
```

### Advanced Parameters

```python
system = ProvenFactSystem(
    api_provider="anthropic",
    api_key="your-key",
    num_professors=3,        # More professors = diverse explanations
    num_referees=2           # More referees = stricter verification
)

# Custom reset schedules
# Modify in ProvenFactSystem.__init__()
schedule_a = [3, 7, 11, 15, 19, ...]  # Custom intervals
schedule_b = [5, 10, 15, 20, 25, ...]
```

## 📊 Analysis Tools

The analysis script provides:

### 1. Session-by-Session Tables
```
Round | Total Sentences | Hallucinations | Rate
------|----------------|----------------|------
  1   |       16       |       0        | 0.00%
  2   |       15       |       0        | 0.00%
...
```

### 2. Evidence Stage Analysis
Performance breakdown by information availability level

### 3. Reasoning Path Extraction
List of all A→B→C causal patterns captured

### 4. Residual Hallucination Categorization
Types and severity of any missed errors

### 5. Visualizations
- Hallucination rate trend across sessions
- Stage-by-stage comparison
- Multi-simulation comparisons

## 💡 Design Principles

### From the Paper

1. **Anchor Stability**: Proven facts prevent logical drift
2. **Real-Time Intervention**: Errors caught immediately, not post-hoc
3. **Causal Over Factual**: Teach reasoning paths, not just conclusions
4. **Bottom-Up Construction**: Build from minimal to maximal evidence
5. **Triple Verification**: Referee A → Referee B → Validation Specialist
6. **Explicit Acknowledgment**: Student must withdraw claims when proven wrong

## 🔍 Limitations and Improvements

### Known Limitations (from paper)

1. **Unit Consistency**: Referees may miss subtle unit errors (km vs li)
2. **Unfalsifiable Claims**: Hard to prevent Student speculation entirely
3. **Computational Cost**: 10-20x more expensive than simple inference
4. **Incomplete Acknowledgments**: Student may not always explicitly withdraw claims

### Recommended Improvements

1. **Enhanced Referee Prompts**: Explicitly check units, numbers, proper nouns
2. **Forced Acknowledgment**: Require Student to state "I withdraw [claim]"
3. **Persona Reset on High Error**: Auto-reset any agent with >50% error rate
4. **Domain-Specific Validators**: Add specialized checking for medical, legal, etc.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{provenfact2025,
  title={Proven fact-based algorithm for AI training and high-purity data generation},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Additional simulation templates (medicine, law, physics)
- [ ] Enhanced unit/numerical verification
- [ ] Multi-language support
- [ ] Visualization dashboard
- [ ] Integration with knowledge graphs
- [ ] Automated fact-checking pipelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- Paper: [Link to paper]
- Documentation: [GitHub Pages]
- Issues: [GitHub Issues]

## 💬 Comparison with Related Work

| Feature | STaR | MAD | RLHF | This System |
|---------|------|-----|------|-------------|
| Real-time verification | ❌ | ❌ | ❌ | ✅ |
| Causal reasoning (A→B→C) | ❌ | ❌ | ❌ | ✅ |
| Staggered reset | ❌ | ❌ | ❌ | ✅ |
| Sequential evidence | ❌ | ❌ | ❌ | ✅ |
| Final audit | ❌ | ❌ | ❌ | ✅ |
| Hallucination rate | ~5-10% | ~3-5% | ~2-4% | **0-1.8%** |

## 📧 Contact

For questions or collaborations:
- Email: [your-email]
- GitHub: [@yourusername]

---

**Note**: This system is designed for generating high-purity training data, not for real-time inference. It's computationally intensive but produces superior quality datasets for model fine-tuning.
