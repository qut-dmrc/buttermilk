Project: Collaborative Assessment Interface (IRC-Style) 
Subject: UI Development Instructions - MINOR ITERATION 

### Overarching Philosophy & Core Principles: 
- Academic Rigor: This is a scientific research project. Accuracy, traceability, and robustness are paramount. We cannot afford mistakes stemming from rushed development. 
- Methodical Approach: Take everything step-by-step. Build, test, validate, and document each component or change before moving on. Prioritize stability over features. 
- Simplicity: Strive for the simplest possible implementation that meets the requirements. Avoid unnecessary complexity. Do not add defensive coding, just raise informative errors. Avoid duplicating code. 
- User Experience: While functional and robust, the interface must also "look amazing" within the defined aesthetic constraints and be intuitive for the user (the academic researcher). The user is an active supervisor of the chat and should feel in control. Interface elements should support, not dominate, their interaction. 
- Core Design Conceit: Retro Terminal Group Chat (IRC-like) metaphor: The user is participating in a group chat session within an interface styled after classic terminal-based IRC clients (e.g., irssi, mIRC with a text-based theme). Think text-based interaction, clear delineation of messages, user lists (in our case, the summary sidebar), and command-line sensibilities. Rationale: This conceit provides a familiar interaction model for text-heavy tasks, focuses attention on the content, and aligns with the desired retro-tech/cyberpunk aesthetic.
- Minimal CSS + JS: We are using Sveltekit. You should always rely on existing libraries, including Bootstrap, wherever possible to reduce the amount of custom code that needs to be maintained.

## Task:

1. Display agent information. 

In the main terminal window and in the Run summary, currently judge messages are being attributed to 'JUDGE-xxxxx'. There is more information available in the agent_info field that we are not using. We don't want too much information, but we should look for and include parameters like 'model' and 'criteria' if they exist. That way the judges will have much more understandable names. I'd like to assign color to judges by the model too, for consistency. I don't really like the cartoony judge emoji icons; let's find a better way to represent each of the groupchat agents that is compact but fits the theme.

2. the run summary sidebar is too wide and individual summary messages are too tall. Let's make them a lot more compact. Put the details, graphic violations indicator on the same line as reasons and scores. Try to make these look a bit more to fit the retro terminal console theme.

3. The 'scorer' part of the run summary sidebar doesn't use space well. There's not much information when the panels are minimised -- it seems like this could be compressed a bit. We probably don't care that much about the names of the scorers ("Scorer YPEBba" is basically gibberish) -- if we color them by model, that would be enough. Maybe we could just get away with displaying the ascii score representation symbols. Then, when the user opens the scorer panel, the ifnormation doesn't actually show -- it says something like:
```
📊 Scorer YPEBba
100%
[object Object]
[object Object]
```
The user should be able to see the full score justifications when they want to.

