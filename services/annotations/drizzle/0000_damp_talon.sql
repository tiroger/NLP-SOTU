CREATE TABLE `review_assignments` (
	`id` text PRIMARY KEY NOT NULL,
	`token_hash` text NOT NULL,
	`reader` text NOT NULL,
	`title` text NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `review_assignments_token_hash_unique` ON `review_assignments` (`token_hash`);--> statement-breakpoint
CREATE TABLE `review_items` (
	`reader` text NOT NULL,
	`item_key` text NOT NULL,
	`assignment_id` text NOT NULL,
	`ordinal` integer NOT NULL,
	`sentiment` text,
	`notes` text DEFAULT '' NOT NULL,
	`revision` integer DEFAULT 0 NOT NULL,
	`updated_at` text,
	PRIMARY KEY(`reader`, `item_key`),
	FOREIGN KEY (`assignment_id`) REFERENCES `review_assignments`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `items_assignment_idx` ON `review_items` (`assignment_id`);