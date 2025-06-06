# Score Pages API Endpoints Specification

This document specifies the API endpoints required for the new `/score/` pages in the Buttermilk frontend.

## 📡 **Required API Endpoints**

### 1. **Individual Record Details**
```
GET /api/records/{record_id}
```

**Parameters:**
- `record_id` (path parameter) - The unique identifier for the record

**Expected Response:**
```json
{
  "id": "string",
  "name": "string", 
  "content": "string",
  "metadata": {
    "created_at": "2024-01-15T10:30:00Z",
    "dataset": "osb|drag|tonepolice",
    "word_count": 156,
    "char_count": 892
  }
}
```

### 2. **Toxicity Scores for Record**
```
GET /api/records/{record_id}/scores
```

**Parameters:**
- `record_id` (path parameter) - The unique identifier for the record

**Expected Response:**
```json
{
  "record_id": "string",
  "off_shelf_results": {
    "GPT-4": {
      "correct": true,
      "score": 0.85,
      "label": "TOXIC|SAFE",
      "confidence": 0.92,
      "model_version": "gpt-4-0613"
    },
    "Claude-3": {
      "correct": false,
      "score": 0.42,
      "label": "SAFE",
      "confidence": 0.78,
      "model_version": "claude-3-sonnet"
    },
    "Gemini": {
      "correct": true,
      "score": 0.78,
      "label": "TOXIC",
      "confidence": 0.85,
      "model_version": "gemini-pro"
    },
    "LLaMA-2": {
      "correct": true,
      "score": 0.91,
      "label": "TOXIC", 
      "confidence": 0.88,
      "model_version": "llama-2-70b"
    }
  },
  "custom_results": {
    "Judge-GPT4": {
      "step": "judge",
      "score": 0.88,
      "model": "gpt-4-0613",
      "template": "toxicity_judge_v1",
      "criteria": "community_guidelines"
    },
    "Judge-Claude": {
      "step": "judge", 
      "score": 0.45,
      "model": "claude-3-sonnet",
      "template": "toxicity_judge_v1",
      "criteria": "community_guidelines"
    },
    "Synth-GPT4": {
      "step": "synth",
      "score": 0.82,
      "model": "gpt-4-0613",
      "template": "synthesis_v1",
      "criteria": "community_guidelines"
    },
    "Synth-Claude": {
      "step": "synth",
      "score": 0.39,
      "model": "claude-3-sonnet", 
      "template": "synthesis_v1",
      "criteria": "community_guidelines"
    }
  },
  "summary": {
    "off_shelf_accuracy": 0.75,
    "custom_average_score": 0.635,
    "total_evaluations": 8,
    "agreement_rate": 0.62
  }
}
```

### 3. **AI Model Responses for Record** *(Optional - for detailed analysis)*
```
GET /api/records/{record_id}/responses
```

**Parameters:**
- `record_id` (path parameter) - The unique identifier for the record
- `include_reasoning` (query parameter, optional) - Include detailed reasoning (default: true)

**Expected Response:**
```json
{
  "record_id": "string",
  "responses": [
    {
      "agent": "Judge-GPT4",
      "type": "judge",
      "model": "gpt-4-0613",
      "content": "This content violates our community guidelines regarding hate speech targeting specific groups.",
      "score": 0.88,
      "reasoning": "The language used contains derogatory terms and promotes harmful stereotypes against the mentioned communities.",
      "criteria_used": "community_guidelines",
      "template": "toxicity_judge_v1",
      "timestamp": "2024-01-15T10:30:00Z",
      "confidence": 0.92,
      "prediction": true
    },
    {
      "agent": "Judge-Claude",
      "type": "judge",
      "model": "claude-3-sonnet", 
      "content": "While the content discusses sensitive topics, it appears to be educational in nature.",
      "score": 0.45,
      "reasoning": "The context suggests academic analysis rather than promoting harmful behavior. The language, while discussing difficult topics, maintains an analytical tone.",
      "criteria_used": "community_guidelines",
      "template": "toxicity_judge_v1", 
      "timestamp": "2024-01-15T10:31:00Z",
      "confidence": 0.78,
      "prediction": false
    }
  ]
}
```

## 🔄 **Enhanced Existing Endpoint**

### **Records List Enhancement**
```
GET /api/records?flow={flow}&include_scores={boolean}
```

**Additional Parameters:**
- `include_scores` (query parameter, optional) - Include summary scores in list view (default: false)

**Enhanced Response (when include_scores=true):**
```json
[
  {
    "id": "record_123",
    "name": "Example Record Name",
    "description": "Brief description...", 
    "summary_scores": {
      "off_shelf_accuracy": 0.75,
      "custom_average": 0.635,
      "total_evaluations": 8,
      "has_detailed_responses": true
    }
  }
]
```

## 🔧 **Implementation Notes**

### **Data Sources:**
- These endpoints should integrate with your existing Buttermilk orchestrator results
- `off_shelf_results` maps to standard toxicity detection API results  
- `custom_results` maps to your Agent/Judge/Synthesizer workflow outputs
- `responses` provides the detailed message content for IRC-style display

### **Performance Considerations:**
- `/api/records/{record_id}/scores` should be the primary endpoint (combines score data)
- `/api/records/{record_id}/responses` can be separate for detailed analysis (larger payload)
- Consider caching for frequently accessed records

### **Error Handling:**
- Return 404 for non-existent record_ids
- Return 422 for invalid flow parameters
- Include meaningful error messages for debugging

### **Frontend Integration:**
The current implementation uses mock data that matches this structure, so once these endpoints are available, integration should be seamless by updating the `fetchRecordData()` function in `/score/[record_id]/+page.svelte`.

## 🎯 **Datasets Supported:**
- `osb` - Oversight Board dataset
- `drag` - Drag Queen vs White Supremacist dataset  
- `tonepolice` - Tone policing detection dataset

## 📝 **Backend Implementation Status:**

**✅ IMPLEMENTED**: Frontend API proxy endpoints are fully implemented and ready for backend integration:
- All three record detail endpoints (`/api/records/{record_id}`, `/scores`, `/responses`)
- Enhanced records list endpoint with `include_scores` parameter
- Complete mock data for development and testing
- Error handling with graceful fallbacks
- Authorization header forwarding
- Environment-based backend URL configuration

**🔧 REQUIRED**: Backend developers need to implement the actual data endpoints at:
- `{BACKEND_API_URL}/api/records/{record_id}`
- `{BACKEND_API_URL}/api/records/{record_id}/scores` 
- `{BACKEND_API_URL}/api/records/{record_id}/responses`
- `{BACKEND_API_URL}/api/records?flow={flow}&include_scores={boolean}`

The frontend will automatically proxy to these backend endpoints when available, falling back to mock data during development.