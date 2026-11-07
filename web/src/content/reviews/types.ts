export type ReviewQuestion = { question: string; answer: string; why: string };

export type ChapterReview = { why: string; takeaways: string[]; questions: ReviewQuestion[] };

export const emptyReview: ChapterReview = { why: "", takeaways: [], questions: [] };
