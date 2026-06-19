module.exports = {
  apps: [
    {
      name: "tp14-paper-sim",
      script: "tp14_paper_sim.py",
      interpreter: process.env.TP14_PYTHON || ".venv/bin/python",
      args: "start --config config/paper_config.json --workers 4",
      autorestart: true,
      max_memory_restart: "1500M",
      time: true,
      out_file: "runs/tp14/pm2.out.log",
      error_file: "runs/tp14/pm2.err.log"
    }
  ]
};
