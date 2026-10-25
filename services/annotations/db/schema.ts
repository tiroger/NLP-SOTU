import { integer, sqliteTable, text, primaryKey, index } from "drizzle-orm/sqlite-core";

export const assignments = sqliteTable("review_assignments", {
  id: text("id").primaryKey(),
  tokenHash: text("token_hash").notNull().unique(),
  reader: text("reader").notNull(),
  title: text("title").notNull(),
  createdAt: text("created_at").notNull(),
});

export const items = sqliteTable("review_items", {
  reader: text("reader").notNull(),
  itemKey: text("item_key").notNull(),
  assignmentId: text("assignment_id").notNull().references(() => assignments.id),
  ordinal: integer("ordinal").notNull(),
  sentiment: text("sentiment"),
  notes: text("notes").notNull().default(""),
  revision: integer("revision").notNull().default(0),
  updatedAt: text("updated_at"),
}, (table) => [primaryKey({columns:[table.reader,table.itemKey]}), index("items_assignment_idx").on(table.assignmentId)]);
