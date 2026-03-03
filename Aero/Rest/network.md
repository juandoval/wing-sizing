graph TD
    A[Geometry Engine] --> B[Mesh Pipeline]
    B --> C[Solver Integration]
    C --> D[Data Platform]
    D --> E[ML Optimization]
    E --> C
    F[Desktop UI] --> G[Backend API]
    G --> A
    G --> B
    G --> C
    G --> D
    G --> E
    H[DevOps] --> G
    I[Validation] --> C
    J[Commercial] --> G