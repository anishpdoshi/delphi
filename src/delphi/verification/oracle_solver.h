#ifndef CPROVER_FASTSYNTH_ORACLE_SOLVER_H
#define CPROVER_FASTSYNTH_ORACLE_SOLVER_H

#include <solvers/decision_procedure.h>
#include <solvers/smt2/smt2_parser.h>

#include <util/mathematical_expr.h>
#include <util/message.h>

#include <delphi/oracle_constraint_gen.h>

#include <unordered_set>
#include <util/cout_message.h>
#include "delphi/synthesis/synthesizer.h"

class oracle_solvert : public decision_proceduret
{
public:
  using oracle_funt = smt2_parsert::oracle_funt;
  using oracle_fun_mapt = smt2_parsert::oracle_fun_mapt;
  bool cache=true;

  const oracle_fun_mapt *oracle_fun_map = nullptr;

  /// Type of oracle representation to synthesize or learn
  enum repr_syntht
    {
        NO_REPR,
        SYGUS_REPR,
        NEURAL_REPR,
        DT_REPR,
        SYMB_REGRESSION_REPR
    };

  struct oracle_repr_optionst {
      repr_syntht repr_type;
      int frequency;
      bool true_false_prediction;
      std::string smtfilepath;
  };

  oracle_solvert(
          decision_proceduret &sub_solver,
          oracle_repr_optionst repr_options,
          message_handlert &);


  // overloads
  void set_to(const exprt &expr, bool value) override;
  exprt handle(const exprt &expr) override;
  exprt get(const exprt &expr) const override;
  void print_assignment(std::ostream &out) const override;
  std::string decision_procedure_text() const override;

  std::size_t get_number_of_solver_calls() const override
  {
    return number_of_solver_calls;
  }

  using oracle_historyt = std::map<std::vector<exprt>, exprt>;
  std::unordered_map<std::string, oracle_historyt> oracle_call_history;

  std::unordered_map<std::string, std::string> oracle_representations;

  exprt get_oracle_value(const function_application_exprt &oracle_app);
  // make call to oracle with single return
  exprt make_oracle_call(const std::string &binary_name, const std::vector<std::string> &argv);

  std::unordered_map<std::string, solutiont> representations;
  std::unordered_map<std::string, problemt> oracle_problem_cache;
  void synth_oracle_representations();
  void learn_oracle_representations();
  void substitute_oracles();

  struct applicationt
  {
    std::string binary_name;
    std::vector<exprt> argument_handles; // arguments
    exprt handle; // result
  };

  using applicationst = std::unordered_map<function_application_exprt, applicationt, irep_hash>;
  applicationst applications;
protected:
  resultt dec_solve() override;

  decision_proceduret &sub_solver;
  oracle_repr_optionst repr_options;
  messaget log;
  std::size_t number_of_solver_calls = 0;
  std::size_t handle_counter = 0;

  using check_resultt = enum { INCONSISTENT, CONSISTENT, ERROR };
  check_resultt check_oracles();
  check_resultt check_oracle(const function_application_exprt &, const applicationt &);
  exprt call_oracle(const applicationt &application, const std::vector<exprt> &inputs);
  exprt call_oracle(const applicationt &application, const std::vector<exprt> &inputs, bool &isnew);
};

#endif // CPROVER_FASTSYNTH_ORACLE_SOLVER_H
