module.exports = {
  apps: [
    {
      name: "tp14-paper-sim",
      script: process.env.TP14_PYTHON || ".venv/bin/python",
      interpreter: "none",
      args: "tp14_paper_sim.py start --config config/paper_config.json --workers 2",
      autorestart: true,
      max_memory_restart: "1500M",
      time: true,
      out_file: "runs/tp14/pm2.out.log",
      error_file: "runs/tp14/pm2.err.log"
    }
  ]
};
