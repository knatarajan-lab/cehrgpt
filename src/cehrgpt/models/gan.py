import torch.nn as nn


class DemographicDiscriminator(nn.Module):
    def __init__(
        self, num_of_genders, num_of_races, num_of_ages, num_of_years, dropout_rate=0.2
    ):
        super(DemographicDiscriminator, self).__init__()
        self.output_dim = num_of_years + num_of_ages + num_of_genders + num_of_races
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.output_dim, 1),
        )

    def forward(self, inputs):
        inputs = inputs.view(-1, self.output_dim)
        return self.classifier(inputs)


class DemographicGenerator(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_genders,
        num_races,
        num_age_groups,
        num_years,
        dropout_rate,
    ):
        super(DemographicGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.initial_year_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_years),
            nn.ReLU(),
        )
        self.initial_age_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_age_groups),
            nn.ReLU(),
        )
        self.gender_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_genders),
            nn.ReLU(),
        )
        self.race_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_races),
            nn.ReLU(),
        )

    def forward(self, x):
        initial_year = self.initial_year_generator(x)
        initial_age = self.initial_age_generator(x)
        gender = self.gender_generator(x)
        race = self.race_generator(x)
        return initial_year, initial_age, gender, race
