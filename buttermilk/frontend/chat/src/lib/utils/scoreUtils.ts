
export function calculateAverageScore(assessmentMessages: Message[]): number | null {
    if (!assessmentMessages || assessmentMessages.length === 0) {
      return null;
    }
    
    let totalScore = 0;
    let validScores = 0;
    
    assessmentMessages.forEach(msg => {
      if (msg.outputs && msg.outputs.correctness !== undefined) {
        const score = parseFloat(msg.outputs.correctness);
        if (!isNaN(score)) {
          totalScore += score;
          validScores++;
        }
      }
    });
    
    return validScores > 0 ? totalScore / validScores : null;
  }