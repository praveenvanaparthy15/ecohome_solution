# EcoHome Energy Advisor

An AI-powered household energy optimization assistant built using **LangChain, LangGraph, Cohere, ChromaDB, SQLite, and LLM-as-a-Judge evaluation**.

The system analyzes household energy usage, solar generation, electricity pricing, weather conditions, and energy-efficiency knowledge to provide practical recommendations for reducing energy consumption and electricity costs.

---

## 1. Problem Statement

Homeowners often have difficulty understanding:

* Which appliances consume the most energy
* When electricity usage is most expensive
* When to charge an EV
* How to optimize HVAC usage
* How to make better use of solar generation
* Whether battery storage may be beneficial
* Which energy-saving actions can provide measurable savings

EcoHome Energy Advisor combines historical household data, retrieval-augmented knowledge, simulated external services, and an AI agent to answer these questions.

---

## 2. Key Capabilities

The application supports:

* Household energy usage analysis
* Device-level energy analysis
* Solar generation analysis
* Date-aware electricity price analysis
* Dynamic weather forecasting
* Weather-aware recommendations
* EV charging optimization
* HVAC optimization
* Smart-home energy management
* Energy-efficiency knowledge retrieval using RAG
* Recent energy usage and solar summary
* Electricity cost and savings estimation
* Multi-step tool-based reasoning using LangGraph
* LLM-as-a-Judge response evaluation
* LLM-as-a-Judge tool usage evaluation
* LLM-generated evaluation recommendations

---

## 3. Architecture

```text
                         User Question
                              |
                              v
                    +--------------------+
                    |   EcoHome Agent    |
                    |      LangGraph     |
                    +---------+----------+
                              |
                       Plan / Select Tools
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
    Energy Usage       Electricity Prices      Weather
       Tool                   Tool              Tool
          |                   |                   |
          +-------------------+-------------------+
                              |
          +-------------------+-------------------+
          |                                       |
          v                                       v
   Solar Generation                       Energy Tips RAG
        Tool                                ChromaDB
          |                                       |
          +-------------------+-------------------+
                              |
                              v
                     Savings Calculator
                              |
                              v
                    Final Recommendation
                              |
                              v
                    LLM-as-a-Judge Evaluation
```

---

## 4. Technology Stack

| Technology | Purpose                          |
| ---------- | -------------------------------- |
| Python     | Application development          |
| LangChain  | LLM and tool integration         |
| LangGraph  | Agent workflow and orchestration |
| Cohere     | LLM and embeddings               |
| ChromaDB   | Vector database for RAG          |
| SQLite     | Historical energy and solar data |
| SQLAlchemy | Database ORM                     |
| Pandas     | Data analysis and evaluation     |
| Matplotlib | Optional visualization           |
| Jupyter    | Experimentation and evaluation   |

---

## 5. Project Structure

```text
ecohome_solution/

│
├── models/
│   ├── __init__.py
│   └── energy.py
│
├── data/
│   ├── documents/
│   │   ├── tip_energy_savings.txt
│   │   ├── tip_device_best_practices.txt
│   │   ├── tip_hvac_optimization.txt
│   │   ├── tip_smart_home_automation.txt
│   │   ├── tip_renewable_energy.txt
│   │   ├── tip_seasonal_energy.txt
│   │   ├── tip_energy_storage.txt
│   │   └── tip_ev_optimization.txt
│   │
│   ├── energy_data.db
│   └── vectorstore/
│
├── agent.py
├── tools.py
├── requirements.txt
│
├── 01_db_setup.ipynb
├── 02_rag_setup.ipynb
├── 03_run_and_evaluate.ipynb
│
├── evaluation_test_results.csv
├── evaluation_scores.csv
├── evaluation_report.json
│
└── README.md
```

---

## 6. Database

The SQLite database contains two primary entities.

### EnergyUsage

Stores historical household energy consumption.

Key fields:

* Timestamp
* Device type
* Energy consumption in kWh
* Cost
* Usage period

### SolarGeneration

Stores solar generation information.

Key fields:

* Timestamp
* Generation in kWh
* Weather condition

The database is created and populated using:

```text
01_db_setup.ipynb
```

---

## 7. RAG Knowledge Base

The project uses **ChromaDB** for retrieval-augmented generation.

The knowledge base contains energy-efficiency information covering:

* Energy-saving best practices
* Device optimization
* HVAC optimization
* Smart-home automation
* Renewable energy
* Seasonal energy management
* Energy storage
* EV optimization

The RAG setup is implemented in:

```text
02_rag_setup.ipynb
```

### RAG Flow

```text
Documents
    |
    v
Text Loading
    |
    v
RecursiveCharacterTextSplitter
    |
    v
Cohere Embeddings
    |
    v
ChromaDB
    |
    v
Similarity Search
    |
    v
Relevant Energy Knowledge
    |
    v
Agent Recommendation
```

### Document Chunking

The `search_energy_tips` tool uses `RecursiveCharacterTextSplitter` before creating the ChromaDB vector store.

Documents are split into manageable chunks with overlap to improve retrieval quality and prevent entire documents from being indexed as single records.

```text
chunk_size = 1000
chunk_overlap = 150
```

### Conditional Vector Store Handling

The `search_energy_tips` tool checks whether the ChromaDB vector store already exists.

* If the vector store exists, it is loaded.
* If it does not exist, documents are loaded and split into chunks.
* The chunks are embedded using Cohere embeddings.
* The resulting vectors are stored and persisted in ChromaDB.

This allows the tool to work both with an existing vector store and in a fresh project environment.

---

## 8. Agent and Tools

The LangGraph agent is implemented in:

```text
agent.py
```

The tools are implemented in:

```text
tools.py
```

### Available Tools

The agent provides seven tools:

1. `get_weather_forecast`
2. `get_electricity_prices`
3. `get_energy_usage`
4. `get_solar_generation`
5. `get_recent_energy_summary`
6. `search_energy_tips`
7. `calculate_savings`

The agent dynamically selects and executes relevant tools based on the user's question.

### Recent Energy Summary

`get_recent_energy_summary` combines recent:

* Household energy usage
* Solar generation
* Net grid energy
* Estimated cost
* Average usage
* Average solar generation

This provides a consolidated view of recent household energy performance.

---

## 9. Dynamic Weather and Electricity Pricing

The project uses simulated external services while avoiding identical hardcoded results for every date.

### Dynamic Weather

`get_weather_forecast` derives weather characteristics from the requested date using a seasonal mathematical calculation.

The calculation dynamically varies:

* Temperature
* Cloud cover
* Weather condition
* Solar generation factor

Therefore, different dates can produce different simulated weather conditions.

### Date-Aware Electricity Pricing

`get_electricity_prices` applies a date-derived multiplier to base time-of-use rates.

The resulting:

* Off-peak rate
* Standard rate
* Peak rate

can vary depending on the requested date.

This provides date-dependent behavior while keeping the application self-contained.

---

## 10. Mock External Services

To keep the project self-contained, external weather and electricity pricing services are simulated.

The mock services provide:

* Weather conditions
* Temperature
* Cloud cover
* Solar generation factor
* Time-of-use electricity prices

The simulated values are date-aware rather than identical static responses for every request.

These services can later be replaced with production APIs.

---

## 11. Savings Calculation

The project includes a savings calculation tool that compares current and optimized electricity rates.

Example:

```text
Energy usage:       10 kWh
Current rate:       $0.25/kWh
Optimized rate:     $0.08/kWh

Current cost:       $2.50
Optimized cost:     $0.80
Potential savings:  $1.70
Savings percentage: 68%
```

---

## 12. Evaluation

The evaluation notebook is:

```text
03_run_and_evaluate.ipynb
```

The project contains **12 test cases** covering:

* Peak-hour cost optimization
* Highest-consuming devices
* EV charging
* HVAC optimization
* Solar generation
* Weather-based recommendations
* Savings calculations
* Battery storage
* Smart-home automation
* Appliance usage analysis
* Recent household energy summaries
* Energy-efficiency knowledge retrieval

Each test case contains:

* User question
* Expected tools
* Expected response / ground truth
* Actual response
* Actual executed tool usage
* Evaluation feedback

### LLM-as-a-Judge

The evaluation framework uses Cohere as an **LLM-as-a-Judge** rather than relying only on keyword or arithmetic heuristics.

### Response Evaluation

Responses are evaluated against the `expected_response` ground truth using:

* Accuracy
* Relevance
* Completeness
* Usefulness

Each metric is scored from **1 to 5**.

### Tool Usage Evaluation

The agent's actual LangGraph messages and tool calls are evaluated against the expected tools using:

* Tool Appropriateness
* Tool Completeness

The evaluation tracks the tools actually executed during the agent workflow rather than relying only on declared or expected tool lists.

### Suite-Level Tool Coverage

The evaluation verifies that all seven required tool capabilities are demonstrated across the complete test suite.

It reports:

* Expected-tool coverage
* Actual executed-tool coverage
* Suite-level tool coverage
* Missing required tools

### Evaluation Report

The final evaluation report uses the LLM to generate:

* Strengths
* Weaknesses
* Recommendations

The report display logic is separated into:

```text
display_evaluation_report()
```

---

## 13. Latest Evaluation Results

The latest evaluation completed successfully across all 12 test cases.

| Metric                            |         Result |
| --------------------------------- | -------------: |
| Tests                             |             12 |
| Successful Tests                  |             12 |
| Success Rate                      |           100% |
| Overall Score                     | **29.42 / 30** |
| Overall Percentage                |     **98.07%** |
| Accuracy                          |   **4.92 / 5** |
| Relevance                         |   **5.00 / 5** |
| Completeness                      |   **4.92 / 5** |
| Usefulness                        |   **5.00 / 5** |
| Tool Appropriateness              |   **4.83 / 5** |
| Tool Completeness                 |   **4.75 / 5** |
| Required Tool Capabilities        |          **7** |
| Tool Capabilities Demonstrated    |          **7** |
| Suite Tool Coverage               |       **100%** |
| End-to-End Executed-Tool Coverage |       **100%** |
| Missing Required Tools            |       **None** |

The evaluation demonstrates strong response quality, tool selection, tool execution, and end-to-end coverage across the test suite.

---

## 14. Evaluation Artifacts

The project generates:

```text
evaluation_test_results.csv
evaluation_scores.csv
evaluation_report.json
```

These artifacts contain detailed test execution results, evaluation scores, tool usage, feedback, strengths, weaknesses, and LLM-generated recommendations.

---

## 15. Running the Project

### Create Virtual Environment

```powershell
py -m venv .venv
```

### Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Configure Cohere

Create a `.env` file:

```text
COHERE_API_KEY=your_cohere_api_key
```

Do not commit `.env` to GitHub.

### Run Database Setup

Open and run:

```text
01_db_setup.ipynb
```

### Run RAG Setup

Open and run:

```text
02_rag_setup.ipynb
```

### Run Evaluation

Open and run:

```text
03_run_and_evaluate.ipynb
```

Running the evaluation notebook executes all 12 scenarios and generates the evaluation artifacts.

---

## 16. Agent Workflow

The LangGraph workflow follows this pattern:

```text
START
  |
  v
Plan / Select Tools
  |
  v
Execute Tools
  |
  +------> Additional Tool Calls
  |
  v
Finalize Response
  |
  v
END
```

The agent can perform multiple tool calls before generating the final recommendation.

The workflow maintains:

* Conversation messages
* Executed tool usage
* Errors
* Final response

This information is also used by the evaluation framework.

---

## 17. Limitations

The current implementation uses:

* Simulated weather data
* Simulated electricity pricing
* Sample historical household data
* Static energy-efficiency documents
* Cohere API for LLM and embeddings

The recommendations should therefore be treated as a demonstration rather than a production household energy management system.

---

## 18. Future Enhancements

Potential improvements include:

* Integration with real weather APIs
* Integration with real electricity pricing APIs
* Smart-meter integration
* Real-time IoT/device data
* Personalized household profiles
* Energy consumption forecasting
* Solar generation forecasting
* Advanced RAG with reranking
* Energy usage visualizations
* Production monitoring and alerting
* More sophisticated tool-selection policies
* Improved multi-tool planning
* Real-time household energy optimization

---

## 19. Conclusion

EcoHome Energy Advisor demonstrates how **LangGraph, LangChain, Cohere, RAG, ChromaDB, SQLite, and tool-based AI agents** can be combined to build a practical household energy optimization assistant.

The system successfully completed **12/12 evaluation scenarios with a 100% execution success rate** and achieved a **98.07% LLM-as-a-Judge evaluation score**.

The project demonstrates:

* Dynamic external-service simulation
* Retrieval-augmented energy knowledge
* Recursive document chunking for RAG
* Conditional ChromaDB initialization and loading
* Seven specialized energy tools
* Multi-step LangGraph agent execution
* Actual executed-tool tracking
* Ground-truth-based response evaluation
* LLM-based tool-usage evaluation
* Suite-level verification of all seven tool capabilities
* LLM-generated evaluation recommendations

The project provides an auditable proof-of-concept for AI-assisted household energy optimization.
