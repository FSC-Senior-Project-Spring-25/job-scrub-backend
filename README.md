# Job Scrub: Backend API

![Scrubby Mascot](https://github.com/FSC-Senior-Project-Spring-25/job-scrub-frontend/blob/main/public/assets/Scrubby-logo.gif)

Welcome to the **Job Scrub Backend** - the intelligent engine powering our AI-driven job search platform.

## Architecture Overview

The Job Scrub backend is built on a modern Python-based stack designed for AI-powered job search and career advancement. Our architecture follows a service-oriented approach with dependency injection, making the system modular, testable, and maintainable.

## Core Technologies

- **FastAPI**: High-performance web framework for building APIs with automatic OpenAPI documentation
- **Python 3.11+**: Latest language features for efficient development
- **Google Gemini**: Primary LLM provider via google-generativeai and langchain-google-genai
- **Groq**: Alternative LLM provider for faster inference via langchain-groq
- **LangChain**: Framework for building LLM-powered applications
- **LangGraph**: Agent orchestration framework for complex, multi-step reasoning
- **Firebase Admin SDK**: Database management and user verification
- **AWS S3 (Boto3)**: Document storage and management for resumes and job descriptions
- **Pinecone**: Vector database for semantic search and embeddings
- **Pydantic**: Data validation and settings management
- **PyTorch & Transformers**: Machine learning capabilities and model serving
- **Dependency Injection**: Using fastapi-injectable for clean service management

## Dependency Injection System

Job Scrub uses `fastapi-injectable` to implement a clean dependency injection pattern:

```python
@injectable
```

This approach provides several benefits:
- Clear separation of concerns
- Simpler testing through mock dependencies
- Centralized service management
- Automatic dependency resolution

## Agent Architecture

Our system implements a hierarchical agent architecture powered by LangGraph for intelligent job search and resume optimization. The overall system follows the ReACT (Reasoning and Acting) pattern, allowing agents to think, reason, and use tools to accomplish complex tasks.

### Supervisor Agent

```
+-----------+  
| __start__ |  
+-----------+  
      *        
      *        
      *        
 +---------+   
 | analyze |   
 +---------+   
      *        
      *        
      *        
 +---------+   
 | execute |   
 +---------+   
      *        
      *        
      *        
  +--------+   
  | update |   
  +--------+   
      *        
      *        
      *        
 +---------+   
 | __end__ |   
 +---------+   
```

The Supervisor Agent acts as the central coordinator for all sub-agents. It:
- Analyzes user queries to determine intent
- Selects and delegates to appropriate specialized agents
- Aggregates results from multiple agents when needed
- Maintains context across multiple user interactions
- Provides consistent and coherent responses

### Resume Matching Agent

```
     +-----------+       
     | __start__ |       
     +-----------+       
            *            
            *            
            *            
  +------------------+   
  | extract_keywords |   
  +------------------+   
            *            
            *            
            *            
+---------------------+  
| compute_match_score |  
+---------------------+  
            *            
            *            
            *            
      +---------+        
      | __end__ |        
      +---------+        
```

The Resume Matching Agent specializes in comparing resumes to job descriptions:
- Extracts key requirements and skills from job descriptions
- Identifies skills and experiences in the user's resume
- Computes a match percentage score
- Provides specific feedback on missing qualifications
- Suggests targeted improvements to increase match score
- Uses semantic understanding to identify relevant experience even when keywords differ

### Resume Enhancer Agent

```
        +-----------+         
        | __start__ |         
        +-----------+         
              *               
              *               
              *               
          +-------+           
          | think |           
          +-------+           
         .         .          
       ..           ..        
      .               .       
+-------+         +---------+ 
| tools |         | __end__ | 
+-------+         +---------+ 
```

The Resume Enhancer Agent helps users improve their resumes:
- Analyzes resume structure, content, and formatting
- Identifies weak points and optimization opportunities
- Suggests stronger action verbs and quantifiable achievements
- Recommends industry-specific keywords to include
- Provides guidance on ATS optimization
- Offers section-by-section improvement recommendations

### User Profile Agent

```
        +-----------+         
        | __start__ |         
        +-----------+         
              *               
              *               
              *               
          +-------+           
          | think |           
          +-------+           
         .         .          
       ..           ..        
      .               .       
+-------+         +---------+ 
| tools |         | __end__ | 
+-------+         +---------+ 
```

The User Profile Agent manages user data and career profiles:
- Builds comprehensive skill profiles from resume data
- Tracks user preferences and job search history
- Recommends skill development opportunities
- Maintains career goals and advancement paths
- Integrates with job search to refine recommendations
- Provides personalized career development insights

### Job Search Agent

```
        +-----------+         
        | __start__ |         
        +-----------+         
              *               
              *               
              *               
          +-------+           
          | think |           
          +-------+           
         .         .          
       ..           ..        
      .               .       
+-------+         +---------+ 
| tools |         | __end__ | 
+-------+         +---------+ 
```

The Job Search Agent facilitates intelligent job discovery:
- Processes natural language job search queries
- Translates user requirements into structured search parameters
- Filters and ranks job listings based on user preferences
- Provides insights on job market trends
- Identifies high-potential opportunities based on user profile
- Allows for complex, multi-parameter searches

### User Search Agent

```
        +-----------+         
        | __start__ |         
        +-----------+         
              *               
              *               
              *               
          +-------+           
          | think |           
          +-------+           
         *         .          
       **           ..        
      *               .       
+-------+         +---------+ 
| tools |         | __end__ | 
+-------+         +---------+ 
```

The User Search Agent enables community networking:
- Helps users find others with similar career interests
- Identifies potential mentors in specific fields
- Suggests connections based on skill complementarity
- Facilitates community engagement and collaboration
- Respects privacy preferences and data sharing settings
- Provides industry-specific networking recommendations

## LLM Integration Architecture

Our system uses a layered approach to LLM integration:

1. **Base LLM Layer**: Abstracts provider-specific implementations (Gemini, Groq)
2. **Agent Layer**: Implements ReACT (Reasoning Action) agents using LangGraph
3. **Service Layer**: Connects agents to business logic and data stores
4. **API Layer**: Exposes functionality through FastAPI endpoints

The base Agent class provides:
- Context management
- Tool registration and execution
- State persistence
- Error handling
- Consistent output formatting

## API Routes

Based on our FastAPI application structure:

- `/job/*` - Job listing operations and search
- `/resume/*` - Resume management, parsing, and analysis
- `/chat/*` - Scrubby chatbot conversation endpoints
- `/posts/*` - Community posts and discussions
- `/users/*` - User profiles, search, and follow functionality

Each endpoint is fully documented in the OpenAPI documentation available at `/docs` when running the API.

## Deployment

The backend is deployed on Render for:
- Continuous deployment from GitHub
- Automatic scaling based on traffic
- Monitoring and logging
- SSL/TLS encryption
- Global CDN distribution

We also leverage:
- AWS S3 for document storage
- Pinecone for vector search
- Firebase for authentication and database
- Huggingface for text embedding

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Farmingdale State College Senior Project Team (Spring '25)
- Google for Gemini AI integration
- Groq for high-performance inference
- LangChain and LangGraph teams for agent frameworks
- Huggingface for free open source text embedding models
