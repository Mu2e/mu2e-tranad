"""Reading: fetch DQM metrics from the DQM MCP server and turn them into a clean time series.

  dqm_mcp          - minimal MCP client for the DQM server (standard library only)
  export_from_dqm  - command line: save a DQM source as a wide run/subrun CSV
  filter_from_dqm  - drop constant or uninformative columns
  ingestion        - load the filtered CSV, validate timestamps, handle timezones
"""
