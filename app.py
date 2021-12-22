from flask import Flask

def create_app():
    app = Flask(__name__)

    from routes import main
    app.register_blueprint(main.bp)

    return app

if __name__ == "__main__":
  app = create_app()
  app.run()