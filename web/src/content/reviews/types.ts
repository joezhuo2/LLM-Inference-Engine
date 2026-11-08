export type ReviewQuestion = { question: string; answer: string; why: string };

export type ChapterReview = { lesson: string; why: string; takeaways: string[]; questions: ReviewQuestion[] };

export const emptyReview: ChapterReview = { lesson: "", why: "", takeaways: [], questions: [] };
