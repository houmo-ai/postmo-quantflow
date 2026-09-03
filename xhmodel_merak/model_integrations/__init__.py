"""Built-in model integrations: lightweight declarations and family-owned implementations.

Discovery imports plugin modules only; each registers its recipes, artifact
contracts and semantic adapter names into a private catalog before freeze.
Execution imports the selected implementation lazily. New families do not
edit public build/inference dispatch tables. Shared types own genuinely shared
contracts; model semantics and runtimes remain in the underlying model packages.

Config resolvers perform lightweight, family-specific planning checks. Providers
do not own output locks, publication or file inventory enforcement.
"""
