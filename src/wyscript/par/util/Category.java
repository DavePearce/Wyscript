package wyscript.par.util;

public enum Category {
	/**
	 * Nested loop with only one parallel-for statement
	 */
	GPU_IMPLICIT_NESTED,
	GPU_IMPLICIT_NONNESTED,
	GPU_PART_IMPLICIT_NONNESTED,
	GPU_PART_IMPLICIT_NESTED,
	GPU_NOTALOOP,
	CPU, GPU_EXPLICIT_NESTED
}