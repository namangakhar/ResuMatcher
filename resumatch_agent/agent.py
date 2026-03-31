import os
import pg8000
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.tool_context import ToolContext

load_dotenv()

model_name = os.getenv("MODEL", "gemini-2.5-flash")

# --- Database connection helper ---
def get_db_connection():
    return pg8000.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "resumatch"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )


# --- Tool: Save a profile to the database ---
def save_profile(tool_context: ToolContext, name: str, resume_text: str) -> dict:
    """Saves or updates a candidate's resume in the database for future use.

    Args:
        tool_context: The tool context for state management.
        name: The candidate's full name (used as unique identifier).
        resume_text: The candidate's full resume or LinkedIn profile text.

    Returns:
        A confirmation that the profile has been saved.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO profiles (name, resume_text, updated_at)
               VALUES (%s, %s, CURRENT_TIMESTAMP)
               ON CONFLICT (name)
               DO UPDATE SET resume_text = EXCLUDED.resume_text,
                             updated_at = CURRENT_TIMESTAMP""",
            (name.strip().lower(), resume_text),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "message": f"Profile for '{name}' saved successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save profile: {str(e)}"}


# --- Tool: Retrieve a profile from the database ---
def get_profile(tool_context: ToolContext, name: str) -> dict:
    """Retrieves a candidate's stored resume from the database by name.

    Args:
        tool_context: The tool context for state management.
        name: The candidate's name to look up.

    Returns:
        The stored resume text or a not-found message.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT resume_text FROM profiles WHERE name = %s",
            (name.strip().lower(),),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            return {"status": "found", "resume_text": row[0]}
        else:
            return {"status": "not_found", "message": f"No profile found for '{name}'. Please provide the resume text."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to retrieve profile: {str(e)}"}


# --- Tool: Save resume and JD to shared state for the pipeline ---
def save_resume_and_jd(tool_context: ToolContext, resume_text: str, job_description: str) -> dict:
    """Saves the user's resume and job description into shared state for downstream agents.

    Args:
        tool_context: The tool context for state management.
        resume_text: The candidate's full resume or LinkedIn profile text.
        job_description: The complete job description to match against.

    Returns:
        A confirmation that both inputs have been stored.
    """
    tool_context.state["RESUME"] = resume_text
    tool_context.state["JD"] = job_description
    return {"status": "success", "message": "Resume and JD saved. Starting analysis."}


# --- Agent 1: Analyzer - computes match score, strengths, gaps ---
analyzer_agent = Agent(
    name="analyzer",
    model=model_name,
    description="Analyzes resume against job description and produces a detailed match report.",
    instruction="""You are a resume-JD matching expert. Analyze the RESUME against the JD thoroughly.

Produce a detailed analysis in EXACTLY this format:

## 📊 Match Analysis

**Overall Match Score:** [X/100]

**Verdict:** [Strong Match / Moderate Match / Weak Match - brief explanation]

### ✅ Strengths (Skills & Experience that Align)
- [Specific strength referencing actual resume content and JD requirement]
- [Continue for all matches found]

### ⚠️ Gaps (What the JD Requires but Resume Lacks)
- [Specific gap — with concrete suggestion to address it]
- [Continue for all gaps found]

### 🔑 ATS Keyword Suggestions
Keywords from the JD to add to the profile:
- [Keyword 1]
- [Keyword 2]
- [Continue...]

Rules:
- Reference SPECIFIC skills and experiences from the resume, not generic statements.
- Be honest with the score. Don't inflate it.
- Every gap must have an actionable suggestion.

RESUME:
{RESUME}

JD:
{JD}
""",
    output_key="match_analysis",
)


# --- Agent 2: Cover Letter Writer ---
cover_letter_agent = Agent(
    name="cover_letter_writer",
    model=model_name,
    description="Generates a tailored cover letter based on the resume, JD, and match analysis.",
    instruction="""You are a professional cover letter writer. Using the candidate's resume, the job description,
and the match analysis, write a polished, ready-to-send cover letter.

The cover letter MUST:
- Be 3-4 paragraphs
- Open with genuine enthusiasm for the specific role (not generic)
- Highlight the candidate's strongest matching skills with concrete examples from their resume
- Acknowledge growth areas as opportunities, framing them positively
- Close with a confident call to action
- Sound human and natural — NOT like a template
- Use professional but warm tone

Format it as:

## 📝 Tailored Cover Letter

Dear Hiring Manager,

[Paragraph 1: Hook — enthusiasm for the role + strongest qualification]

[Paragraph 2: Body — 2-3 specific achievements/skills mapped to JD requirements]

[Paragraph 3: Address gaps positively + cultural fit / motivation]

[Paragraph 4: Confident close + call to action]

Sincerely,
[Candidate]

RESUME:
{RESUME}

JD:
{JD}

MATCH ANALYSIS:
{match_analysis}
""",
    output_key="cover_letter",
)


# --- Agent 3: Resume Optimizer ---
resume_optimizer_agent = Agent(
    name="resume_optimizer",
    model=model_name,
    description="Rewrites and optimizes the resume specifically tailored for the target JD.",
    instruction="""You are an expert resume writer and ATS optimization specialist.
Rewrite the candidate's resume to be PERFECTLY tailored for the target JD.

You MUST produce a complete, copy-paste ready resume in this format:

## 📄 Optimized Resume

### [CANDIDATE NAME]
[Contact info placeholder — Email | Phone | LinkedIn | Location]

---

#### PROFESSIONAL SUMMARY
[3-4 lines. Lead with years of experience + core expertise. Include top JD keywords naturally.
Frame everything toward the target role.]

#### TECHNICAL SKILLS
[Reorder to prioritize JD-relevant skills FIRST. Add suggested skills in brackets like [AWS - suggested].
Group by category: Languages, Frameworks, Cloud/DevOps, Databases, Tools]

#### PROFESSIONAL EXPERIENCE

**[Job Title]** | [Company] | [Dates]
- [Rewrite each bullet to emphasize JD-relevant achievements]
- [Use action verbs + metrics where possible]
- [Weave in JD keywords naturally]

#### EDUCATION
[Degree, University, Year]

#### CERTIFICATIONS (Suggested)
[List 2-3 relevant certifications the candidate should pursue based on gaps]

---

Rules:
- NEVER fabricate experience or skills the candidate doesn't have
- Mark suggested additions clearly with [suggested] tags
- Reorder and reframe existing content to maximize JD alignment
- Include ATS keywords from the JD naturally throughout
- Make it immediately copy-pasteable

RESUME:
{RESUME}

JD:
{JD}

MATCH ANALYSIS:
{match_analysis}
""",
    output_key="optimized_resume",
)


# --- Agent 4: Formatter - assembles final polished output ---
formatter_agent = Agent(
    name="formatter",
    model=model_name,
    description="Assembles all analysis into a clean, final output for the user.",
    instruction="""You are the final presenter. Take ALL the work done by previous agents and present it
as one clean, well-organized response. Do NOT re-analyze or change the content — just assemble it beautifully.

Present in this exact order:
1. The Match Analysis (from match_analysis)
2. The Tailored Cover Letter (from cover_letter)
3. The Optimized Resume (from optimized_resume)

Add a brief intro line like:
"Here's your complete ResuMatch analysis! Below you'll find your match report, a tailored cover letter, and an optimized resume — all ready to use. 🚀"

And a closing line like:
"💡 Need changes? Just ask — I can adjust the tone, focus on different skills, or rewrite any section!"

Keep ALL the original formatting, headers, and emojis intact. Don't summarize or truncate.

MATCH ANALYSIS:
{match_analysis}

COVER LETTER:
{cover_letter}

OPTIMIZED RESUME:
{optimized_resume}
""",
)


# --- Sequential Workflow: chains all 4 agents ---
analysis_workflow = SequentialAgent(
    name="analysis_workflow",
    description="Full resume-JD analysis pipeline: analyze → cover letter → optimize resume → format output.",
    sub_agents=[
        analyzer_agent,
        cover_letter_agent,
        resume_optimizer_agent,
        formatter_agent,
    ],
)


# --- Root Agent: Greeter that collects inputs then hands off ---
root_agent = Agent(
    name="resumatch_agent",
    model=model_name,
    description="ResuMatch — AI career advisor that analyzes resumes against job descriptions.",
    instruction="""You are **ResuMatch**, a friendly AI career advisor.

Your job is to collect a resume and a job description, then hand off to the analysis workflow.

## You have 3 tools:
- `get_profile(name)` — Look up a stored resume by name
- `save_profile(name, resume_text)` — Save/update a resume in the database
- `save_resume_and_jd(resume_text, job_description)` — Store both for analysis pipeline

## Conversation flow:

1. Greet the user. Ask if they want to:
   a) **Look up an existing profile** (just provide their name + a JD)
   b) **Submit a new resume** (paste resume text + JD)

2. **If they give a name to look up:**
   - Use `get_profile` to fetch their stored resume.
   - If found, show them a brief confirmation of what's on file, then ask for the JD.
   - If not found, tell them and ask them to paste their resume instead.

3. **If they paste a new resume:**
   - Ask for their name so you can save it.
   - Use `save_profile` to store it in the database for future use.
   - Then ask for the JD.

4. **Once you have BOTH resume text AND JD:**
   - Use `save_resume_and_jd` to store them in state.
   - Transfer to the `analysis_workflow` agent.

## Rules:
- Do NOT analyze anything yourself. Only collect inputs and hand off.
- Do NOT ask for the same thing twice. Once you have it, move on.
- Be concise and friendly.
- If user provides everything in one message (name + JD, or resume + JD), handle it all at once.
""",
    tools=[save_resume_and_jd, save_profile, get_profile],
    sub_agents=[analysis_workflow],
)
