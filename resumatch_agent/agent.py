import os
from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.tool_context import ToolContext

load_dotenv()

model_name = os.getenv("MODEL", "gemini-2.5-flash")


# --- Tool: Save resume to state ---
def save_inputs_to_state(tool_context: ToolContext, resume_text: str, job_description: str) -> dict:
    """Saves the user's resume and job description to state for analysis.

    Args:
        tool_context: The tool context for state management.
        resume_text: The candidate's full resume or LinkedIn profile text.
        job_description: The complete job description to match against.

    Returns:
        A confirmation that inputs have been stored.
    """
    tool_context.state["RESUME"] = resume_text
    tool_context.state["JD"] = job_description
    return {"status": "success"}


# --- Step 1: Analyzer ---
analyzer_agent = Agent(
    name="analyzer",
    model=model_name,
    description="Analyzes resume against JD and produces match report.",
    instruction="""You are a resume-JD matching expert. Analyze the RESUME against the JD.

Produce analysis in EXACTLY this format:

## Match Analysis

**Overall Match Score:** [X/100]
**Verdict:** [Strong Match / Moderate Match / Weak Match - brief explanation]

### Strengths (Skills & Experience that Align)
- [Specific strength referencing actual resume content and JD requirement]

### Gaps (What the JD Requires but Resume Lacks)
- [Specific gap with concrete suggestion to address it]

### ATS Keyword Suggestions
- [Keywords from the JD to add to the profile]

Be specific, honest, and actionable.

RESUME:
{RESUME}

JD:
{JD}
""",
    output_key="match_analysis",
)


# --- Step 2: Cover Letter Writer ---
cover_letter_agent = Agent(
    name="cover_letter_writer",
    model=model_name,
    description="Generates a tailored cover letter.",
    instruction="""Write a professional, ready-to-send cover letter for the candidate.

Requirements:
- 3-4 paragraphs
- Open with enthusiasm for the specific role
- Highlight strongest matching skills with concrete examples
- Frame gaps as growth opportunities
- Close with confident call to action
- Sound human and natural, NOT generic

RESUME:
{RESUME}

JD:
{JD}

MATCH ANALYSIS:
{match_analysis}
""",
    output_key="cover_letter",
)


# --- Step 3: Resume Optimizer ---
resume_optimizer_agent = Agent(
    name="resume_optimizer",
    model=model_name,
    description="Rewrites and optimizes the resume for the target JD.",
    instruction="""Rewrite the candidate's resume tailored for the target JD. Produce a complete, copy-paste ready resume.

Include:
- PROFESSIONAL SUMMARY (3-4 lines with JD keywords)
- TECHNICAL SKILLS (reordered for JD, suggested additions marked with [suggested])
- PROFESSIONAL EXPERIENCE (bullets rewritten to emphasize JD-relevant achievements)
- EDUCATION
- SUGGESTED CERTIFICATIONS (based on gaps)

Rules:
- NEVER fabricate experience the candidate doesn't have
- Mark suggested additions with [suggested]
- Include ATS keywords naturally
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


# --- Step 4: Formatter ---
formatter_agent = Agent(
    name="formatter",
    model=model_name,
    description="Assembles all outputs into one clean response.",
    instruction="""Present ALL the work from previous agents as one clean response.

Start with: "Here's your complete ResuMatch analysis! Below you'll find your match report, a tailored cover letter, and an optimized resume — all ready to use."

Then present in this order:
1. Match Analysis
2. Tailored Cover Letter
3. Optimized Resume

End with: "Need changes? Just ask — I can adjust the tone, focus on different skills, or rewrite any section!"

Keep ALL original formatting intact. Do NOT re-analyze, summarize, or truncate.

MATCH ANALYSIS:
{match_analysis}

COVER LETTER:
{cover_letter}

OPTIMIZED RESUME:
{optimized_resume}
""",
)


# --- Pipeline: chains all 4 agents ---
analysis_pipeline = SequentialAgent(
    name="analysis_pipeline",
    description="Runs the full analysis: match scoring, cover letter, resume optimization, formatting.",
    sub_agents=[
        analyzer_agent,
        cover_letter_agent,
        resume_optimizer_agent,
        formatter_agent,
    ],
)


# --- Root Agent ---
root_agent = Agent(
    name="resumatch_agent",
    model=model_name,
    description="ResuMatch — collects resume and JD, then runs analysis pipeline.",
    instruction="""You are ResuMatch, a friendly AI career advisor.

Your ONLY job is to collect the user's resume and job description, then hand off.

Step 1: Greet the user. Ask them to provide their resume or LinkedIn profile text.
Step 2: When they provide their resume, ask for the job description (JD).
Step 3: Once you have BOTH, call the `save_inputs_to_state` tool with the resume_text and job_description.
        After the tool succeeds, immediately transfer control to the `analysis_pipeline` agent.

IMPORTANT RULES:
- After calling save_inputs_to_state, you MUST transfer to analysis_pipeline. Do NOT respond yourself.
- Do NOT analyze anything. Do NOT write cover letters or resumes. Just collect and hand off.
- If the user gives both resume and JD in one message, call the tool and transfer immediately.
- Never ask for the resume or JD a second time.
""",
    tools=[save_inputs_to_state],
    sub_agents=[analysis_pipeline],
)
