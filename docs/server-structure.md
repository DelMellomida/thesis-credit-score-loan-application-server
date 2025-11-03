C:.
│   .env
│   .gitignore
│   main.py                     # Main FastAPI application entry point
│   requirements.txt
│   docker-compose.yml
│
├───app
│   ├───api
│   │   │   __init__.py
│   │   │   auth.py             # Endpoints for /signup, /login
│   │   │   loan.py             # Endpoints for /predict, /explain
│   │   └───dependencies.py     # FastAPI dependency injection
│   │
│   ├───core
│   │   │   __init__.py
│   │   │   config.py           # Application configuration
│   │   └───security.py         # JWT and password hashing logic
│   │
│   ├───database
│   │   │   __init__.py
│   │   │   connection.py       # MongoDB connection setup
│   │   └───models              # FOLDER for database document models
│   │       │   __init__.py
│   │       │   user_model.py   # Defines the User document structure for MongoDB
│   │       └───loan_model.py   # Defines the Loan Application/Result structure
│   │
│   ├───schemas
│   │   │   __init__.py
│   │   └───schemas.py          # All Pydantic models for API requests/responses
│   │
│   └───services
│       │   __init__.py
│       │   prediction_service.py # Core logic for credit scoring and recommendations
│       └───rag_service.py        # Core logic for the RAG chatbot explanations
│
├───data
│   ├───raw
│   │   └───synthetic_training_data.csv # Your initial, purpose-built dataset
│   │
│   └───knowledge_base
│       │   loan_policies.md
│       │   scoring_factors.md
│       └───financial_tips.md
│
├───models
│   │   credit_model.pkl        # The saved, trained Penalized Logistic Regression model
│   │   encoder.pkl             # The saved OneHotEncoder for categorical features
│   └───scaler.pkl              # The saved StandardScaler for numerical features
│
├───scripts
│   │   train_model.py          # Script to preprocess data and train the model
│   │   retrain_model.py        # Script to export from MongoDB and retrain the model
│   └───setup_rag.py            # One-time script to create the vector database
│
├───tests
│   │   test_auth.py            # Tests for authentication endpoints
│   └───test_loan.py            # Tests for prediction and explanation endpoints
│
└───docs
    │   README.md
    └───API_DOCUMENTATION.md

