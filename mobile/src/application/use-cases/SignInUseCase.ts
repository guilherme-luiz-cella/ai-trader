import type { AuthSession } from "../../domain/entities/AuthSession";
import { normalizeBackendBaseUrl } from "../../domain/value-objects/BackendConfig";
import type { TradingControlRepository } from "../ports/TradingControlRepository";

type Input = {
  backendUrl: string;
  email: string;
  password: string;
};

export class SignInUseCase {
  constructor(private readonly repository: TradingControlRepository) {}

  async execute(input: Input): Promise<AuthSession> {
    const config = normalizeBackendBaseUrl(input.backendUrl);
    const email = input.email.trim().toLowerCase();
    const password = input.password;

    if (!email) {
      throw new Error("Email is required.");
    }

    if (!password) {
      throw new Error("Password is required.");
    }

    return this.repository.signIn(config, {
      email,
      password,
    });
  }
}
