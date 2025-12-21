class CreateProcessingJobs < ActiveRecord::Migration[7.1]
  def change
    create_table :processing_jobs do |t|
      t.integer :status, default: 0, null: false

      t.timestamps
    end
  end
end
